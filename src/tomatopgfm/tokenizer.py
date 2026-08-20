from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DNA_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def rc(seq: str) -> str:
    return seq.translate(DNA_COMP)[::-1].upper()


@dataclass
class RCKmerTokenizer:
    k: int = 6
    vocab: dict[str, int] | None = None

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def mask_id(self) -> int:
        return 1

    @property
    def unk_id(self) -> int:
        return 2

    def train(self, sequences: list[str], vocab_size: int = 4096) -> None:
        counts: Counter[str] = Counter()
        for seq in sequences:
            seq = seq.upper()
            for i in range(max(0, len(seq) - self.k + 1)):
                tok = seq[i : i + self.k]
                if set(tok) <= set("ACGTN"):
                    canon = min(tok, rc(tok))
                    counts[canon] += 1
        vocab = {"[PAD]": 0, "[MASK]": 1, "[UNK]": 2, "[CLS]": 3, "[SEP]": 4}
        for tok, _ in counts.most_common(max(0, vocab_size - len(vocab))):
            if tok not in vocab:
                vocab[tok] = len(vocab)
        self.vocab = vocab

    def canonical(self, kmer: str) -> str:
        """Return the strand-canonical form of a k-mer (min of kmer / its RC).

        Canonicalizing at encode time gives reverse-complement *soft consistency*:
        a k-mer and its reverse complement collapse to the same token id. This is
        intentionally weaker than full RC equivariance (sequence order is not
        reversed), matching the configured rc_policy: soft_consistency_only.
        """
        return min(kmer, rc(kmer))

    def encode_kmers(self, seq: str) -> list[int]:
        """Tile a sequence into non-overlapping canonical k-mer ids (no specials).

        This is the core tokenization used by the real shard builder, where each
        emitted id must line up with the graph node (segment) it came from, so no
        [CLS]/[SEP] are added here. ``encode`` wraps this with the special tokens.
        """
        if self.vocab is None:
            raise RuntimeError("Tokenizer is not trained")
        seq = seq.upper()
        ids: list[int] = []
        for i in range(0, max(0, len(seq) - self.k + 1), self.k):
            kmer = seq[i : i + self.k]
            tok = self.canonical(kmer) if set(kmer) <= set("ACGTN") else kmer
            ids.append(self.vocab.get(tok, self.unk_id))
        return ids

    def encode(self, seq: str, max_len: int | None = None) -> list[int]:
        if self.vocab is None:
            raise RuntimeError("Tokenizer is not trained")
        ids = [self.vocab["[CLS]"], *self.encode_kmers(seq), self.vocab["[SEP]"]]
        if max_len:
            ids = ids[:max_len]
            ids += [self.pad_id] * max(0, max_len - len(ids))
        return ids

    def save(self, path: Path) -> None:
        if self.vocab is None:
            raise RuntimeError("Tokenizer is not trained")
        path.write_text(json.dumps({"k": self.k, "vocab": self.vocab}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RCKmerTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(k=payload["k"], vocab={k: int(v) for k, v in payload["vocab"].items()})

    def audit_rc_symmetry(self) -> dict:
        """Legacy vocab-closure audit.

        Canonical-only tokenizers are expected to fail this check because reverse
        complements collapse to the same canonical id at encode time. Production
        gates should use ``audit_encode_rc_consistency`` and
        ``audit_dead_classes`` instead.
        """
        if self.vocab is None:
            raise RuntimeError("Tokenizer is not trained")
        dna = [t for t in self.vocab if not t.startswith("[")]
        missing = [t for t in dna if rc(t) not in self.vocab]
        return {
            "dna_tokens": len(dna),
            "missing_rc": len(missing),
            "status": "pass" if not missing else "fail",
            "level": "vocab_closure_legacy_not_production_gate",
        }

    def audit_encode_rc_consistency(self, sequences: list[str]) -> dict:
        """Measure frame-sensitive soft RC consistency at encode time.

        For each sequence, the canonical-token multiset of seq and of rc(seq)
        should match. This is only expected to pass when the non-overlapping
        tiling frame is comparable, for example sequences whose length is a
        multiple of k. For production tokenizer gates, use
        ``audit_kmer_rc_consistency``; arbitrary segment lengths are not
        frame-invariant under reverse complement when using non-overlapping
        k-mer tiling.
        """
        if self.vocab is None:
            raise RuntimeError("Tokenizer is not trained")
        special = {self.pad_id, self.mask_id, self.unk_id, self.vocab["[CLS]"], self.vocab["[SEP]"]}
        consistent = 0
        for seq in sequences:
            a = sorted(t for t in self.encode(seq) if t not in special)
            b = sorted(t for t in self.encode(rc(seq)) if t not in special)
            consistent += int(a == b)
        frac = consistent / len(sequences) if sequences else 0.0
        return {
            "sequences": len(sequences),
            "consistent_fraction": frac,
            "status": "pass" if frac == 1.0 else "fail",
            "level": "encode_soft_consistency_frame_sensitive",
        }

    def audit_kmer_rc_consistency(self, kmers: list[str] | None = None) -> dict:
        """Verify each k-mer and its reverse complement map to the same id.

        This is the production RC soft-consistency contract for canonical k-mer
        tokenization. It is independent of segment-length frame effects.
        """
        if self.vocab is None:
            raise RuntimeError("Tokenizer is not trained")
        if kmers is None:
            kmers = [t for t in self.vocab if not t.startswith("[")]
        checked = 0
        mismatches: list[dict[str, object]] = []
        for kmer in kmers:
            kmer = kmer.upper()
            if len(kmer) != self.k or not set(kmer) <= set("ACGTN"):
                continue
            a = self.encode_kmers(kmer)
            b = self.encode_kmers(rc(kmer))
            checked += 1
            if a != b:
                mismatches.append({"kmer": kmer, "rc": rc(kmer), "ids": [a, b]})
                if len(mismatches) >= 20:
                    break
        return {
            "kmers_checked": checked,
            "mismatch_count": len(mismatches),
            "mismatches_preview": mismatches,
            "status": "pass" if not mismatches else "fail",
            "level": "kmer_rc_soft_consistency",
        }

    def audit_dead_classes(self, sequences: list[str]) -> dict:
        """Check whether trained non-special vocab ids are reachable by encoding.

        This is the production class-space audit: every non-special id in the
        vocab should appear at least once when encoding the declared tokenizer
        corpus or a complete smoke-test corpus. Otherwise the MLM head has dead
        classes with no possible positive labels.
        """
        if self.vocab is None:
            raise RuntimeError("Tokenizer is not trained")
        special_tokens = {"[PAD]", "[MASK]", "[UNK]", "[CLS]", "[SEP]"}
        special_ids = {self.vocab[t] for t in special_tokens if t in self.vocab}
        vocab_ids = set(int(v) for k, v in self.vocab.items() if k not in special_tokens)
        emitted: set[int] = set()
        for seq in sequences:
            emitted.update(i for i in self.encode_kmers(seq) if i not in special_ids)
        dead = sorted(vocab_ids - emitted)
        return {
            "vocab_non_special_ids": len(vocab_ids),
            "emitted_non_special_ids": len(emitted),
            "dead_class_count": len(dead),
            "dead_class_ids_preview": dead[:20],
            "status": "pass" if not dead else "fail",
            "level": "dead_class_reachability",
        }
