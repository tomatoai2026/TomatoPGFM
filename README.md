# TomatoPGFM

TomatoPGFM is a graph-conditioned genomic foundation model pretrained on a 66-accession tomato pangenome. It combines reverse-complement-folded non-overlapping 6-mer tokenization, an eight-channel token-aligned graph interface, adjacency-based message passing, bidirectional Mamba2 blocks, and local-attention mixture-of-experts blocks. Five graph-derived channels vary in the production shards, while three interface channels are constant zero.

This repository is the code and evaluation companion to **TomatoPGFM: A graph-conditioned foundation model for tomato pangenomes**. The public Python package, command-line interface and model class use the unified `tomatopgfm` / `TomatoPGFM` naming.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-eval.txt
```

Production checkpoint inference requires a CUDA-compatible installation of
`mamba-ssm`; see `requirements-inference.txt`. A production configuration fails
explicitly when Mamba2 is unavailable. CPU smoke tests request the convolutional
fallback explicitly and do not validate production checkpoint inference.

## Repository layout

- `src/tomatopgfm/`: tokenizer, graph parsing, graph features, shard building, model, training and evaluation utilities.
- `scripts/pretrain.py`: five-stage distributed pretraining entry point.
- `scripts/graph_sensitivity_bootstrap.py`: paired graph-on/shuffle/off analysis.
- `scripts/padding_aware_frozen_probe.py`: padding-aware embedding extraction, fold-wise scaling, nested C selection and bootstrap intervals.
- `scripts/lora_genecds.py`: model-specific LoRA adaptation.
- `scripts/benchmark_efficiency.py`: zero-feature GraphAdapter-on versus graph-off software-path benchmark.
- `configs/`: final architecture and training snapshot.
- `manifests/`: ordered, sequence-free LA1974 and MicroTom evaluation coordinates,
  labels and per-window sequence checksums, plus the public 66-accession manifest.
- `results/`: machine-readable values reported in the manuscript.

## Quick checks

```bash
python -m tomatopgfm.cli static-smoke
python -m tomatopgfm.cli smoke-model
pytest -q
```

## Model weights

The public model release uses an inference-only safetensors file. The full
optimizer-bearing training-resume checkpoint is not included in the public
GitHub, Hugging Face, ModelScope or Zenodo packages. Its identity is retained
only as export provenance in the inference manifest. Verify or strictly load
the public artifact with `scripts/load_inference_weights.py`; see
`MODEL_WEIGHTS.md`. Set
`TOMATOPGFM_CHECKPOINT` to the downloaded `model.safetensors` (the evaluation
scripts also accept a local training checkpoint), `TOMATOPGFM_TOKENIZER` to
`assets/tokenizer_vocab.json`, and `TOMATOPGFM_REPO` to this repository for the
evaluation scripts.

## Reproducing manuscript analyses

Commands and required external inputs are listed in `docs/REPRODUCIBILITY.md`. Exact efficiency-benchmark conditions are documented in `docs/BENCHMARK_CONDITIONS.md`. The frozen-probe result JSON and coordinate-bearing manifests use the padding-aware protocol reported in the manuscript.

The graph-sensitivity bootstrap can be rerun directly from the ten packaged
per-batch NPZ files:

```bash
python scripts/graph_sensitivity_bootstrap.py --batch-all
```

To rebuild sequence-bearing evaluation inputs from a downloaded public FASTA,
use `scripts/materialize_manifest_sequences.py`; the script checks every
reconstructed window against the packaged SHA-256 value.

## Citation and license

Use `CITATION.cff` for citation metadata. The source code in this repository is
licensed under Apache-2.0. Model weights are distributed separately under the
Apache-2.0, as specified in the accompanying model release package.
