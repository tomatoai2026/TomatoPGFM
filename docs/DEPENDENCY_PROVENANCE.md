# Dependency provenance

This file distinguishes historical training provenance from the private
inference-release validation environment.

Historical production records identify PyTorch 2.4 and a CUDA-enabled
`mamba_ssm` environment, but the exact `mamba-ssm` package build was not
captured in the saved training metadata. `requirements-inference.txt` therefore
records a compatible version range rather than claiming an exact historical
training lock.

For the v0.1.0 inference artifact, a private validation run used container
`tomatopgfm-v1-gpu:20260817` (image ID
`sha256:20de0ccf75978eb63b299e4129a5a5d893d2397c3e7ca91a1351844fd0c1ee74`),
PyTorch `2.11.0+cu130`, CUDA runtime `13.0`, an NVIDIA L20 GPU and the
installed `mamba_ssm` package. Strict CUDA loading and finite graph-off and
zero-feature GraphAdapter-on forward checks passed; the full evidence is in
`08_validation_reports/REMOTE_L20_WEIGHT_VALIDATION_20260820.md`. Each public
platform still requires a fresh anonymous download and hash verification.

The external baselines were loaded from:

- `zhihan1996/DNABERT-2-117M`
- `zhangtaolab/plant-dnamamba-BPE`

The exact snapshot revisions were not saved in the historical result JSON.
Future reruns must pin explicit revisions and record them with the results.
