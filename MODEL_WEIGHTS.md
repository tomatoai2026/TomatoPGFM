# TomatoPGFM model weights

Model weights are distributed outside Git.

## Recommended inference release

- File: `model.safetensors`
- Contents: inference-only FP32 model tensors; no optimizer, scheduler or RNG state
- File size: 1,916,843,008 bytes
- Serialized tensors: 604
- Unique parameters: 479,195,678
- Tied-weight representation: `mlm_head.weight` is omitted and restored as an
  alias of `embed.weight` by `scripts/load_inference_weights.py`
- Hugging Face target record (publish and verify before making public): `https://huggingface.co/tomatoai2026/TomatoPGFM`
- ModelScope target record (publish and verify before making public): `https://modelscope.cn/models/turgun/TomatoPGFM`
- Permanent Zenodo model record: DOI to be added after the model record is reserved and published
- SHA-256: `462f3bfc7178c4558fe993ef579d925da6a54a7448c975b8488bf74d0ac36f7c`

## Training-resume checkpoint

The 5,746,314,884-byte optimizer-bearing training-resume checkpoint is **not
distributed in the public release**. It is not included in GitHub, Hugging Face,
ModelScope or Zenodo model/data uploads. The inference manifest retains its
source-checkpoint identity for export provenance only; it is not a download
link or a public archive record.

Verify the inference artifact without instantiating the model:

```bash
PYTHONPATH=src python scripts/load_inference_weights.py \
  --weights model.safetensors --config configs/model_final.yaml --verify-only
```

Remove `--verify-only` and select `--device cuda` to perform strict model
loading in an environment with the production inference dependencies.
