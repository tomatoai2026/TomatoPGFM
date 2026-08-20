# TomatoPGFM model weights

The v0.1.0 inference release is distributed as a tensor-only safetensors
artifact outside the Git source tree.

## Inference release v0.1.0

- File: `model.safetensors`
- Contents: inference-only FP32 model tensors; no optimizer, scheduler or RNG state
- File size: 1,916,843,008 bytes
- Serialized tensors: 604
- Unique parameters: 479,195,678
- Tied-weight representation: `mlm_head.weight` is omitted and restored as an
  alias of `embed.weight` by `scripts/load_inference_weights.py`
- SHA-256: `462f3bfc7178c4558fe993ef579d925da6a54a7448c975b8488bf74d0ac36f7c`

### Permanent records and platforms

- Zenodo model Version DOI: https://doi.org/10.5281/zenodo.22032734
- Zenodo model Concept DOI: https://doi.org/10.5281/zenodo.22032733
- Processed evaluation resources: https://doi.org/10.5281/zenodo.22033079
- GitHub software Version DOI: https://doi.org/10.5281/zenodo.22035724
- GitHub software Concept DOI: https://doi.org/10.5281/zenodo.22035723
- GitHub release: https://github.com/tomatoai2026/TomatoPGFM/releases/tag/v0.1.0
- Hugging Face: https://huggingface.co/tomatoai2026/TomatoPGFM
- ModelScope: https://modelscope.cn/models/turgun/TomatoPGFM

The model weights are released under Apache-2.0. The processed evaluation
resources are archived separately under CC BY 4.0.

## Training-resume checkpoint

The 5,746,314,884-byte optimizer-bearing training-resume checkpoint is **not
distributed in the public release**. It is not included in GitHub, Hugging Face,
ModelScope or Zenodo model/data uploads. The inference manifest retains its
source-checkpoint identity for export provenance only; it is not a download
link or a public archive record.

Verify the public inference artifact without instantiating the model:

```bash
PYTHONPATH=src python scripts/load_inference_weights.py \
  --weights model.safetensors --config configs/model_final.yaml --verify-only
```

Remove `--verify-only` and select `--device cuda` to perform strict model
loading in an environment with the production inference dependencies.
