# Reproducibility guide

## Final configuration

`configs/model_final.yaml` contains the checkpoint-compatible architecture. `configs/pretraining_final.json` records the active objectives, five curriculum stages, optimization schedule, world size, per-rank batch size and gradient accumulation. The original absolute shard paths have been replaced by repository-relative logical paths.

## Graph and pretraining data

Use `tomatopgfm build-pansn`, `tomatopgfm graph-report`, and `tomatopgfm build-pan66-shard-pilot` to inspect assemblies, an rGFA graph and token-aligned graph features. Full pretraining requires the graph, segment table, link table, feature table and frozen tokenizer described in `docs/DATA_SOURCES.md`. The GraphAdapter interface has eight channels; five vary in the production shards and three are constant zero.

The 249-GiB curriculum shards are not redistributed. Their construction is
documented by the public accession manifest, graph and shard builders,
configuration snapshots and source checksums.

## Five-stage pretraining

```bash
export TOMATOPGFM_SHARD_ROOT=/path/to/shards
torchrun --nproc_per_node=4 scripts/pretrain.py   --model-yaml configs/model_final.yaml   --train-yaml configs/train_final.yaml   --shard-root "$TOMATOPGFM_SHARD_ROOT"   --out-root runs
```

The graph-sensitivity analysis can be rerun from the ten packaged paired
per-batch NPZ files with:

```bash
python scripts/graph_sensitivity_bootstrap.py --batch-all
```

The fixed bootstrap seed (`1234`; the graph-shuffle branch uses `1235`) is
recorded in `results/graph_sensitivity/RUN_METADATA.json`. The stage report's
`training_panel` entries refer to the two strata of the 66-accession training
panel, not to LA1974 or MicroTom external-genome evaluations. The coordinate
manifest metadata retains source checksums and public reconstruction roles; the
original source JSONL files are not redistributed.

## Frozen probe

The ordered coordinate manifests are compressed JSONL files in `manifests/`.
Materialize their sequence fields from the cited external FASTA before embedding
extraction. Provide the external embedding NPZ, checkpoint and source-data root
to `scripts/padding_aware_frozen_probe.py`. The script implements mean pooling
over 87 non-padding positions, the 512-position and direct-unpadded sensitivity
modes, fold-wise `StandardScaler`, three-fold C selection and 1,000
class-stratified bootstrap resamples. The baseline embedding NPZ is not included,
so a full three-model frozen-probe rerun requires the two cited baseline models.

## LoRA and efficiency

Set `TOMATOPGFM_CHECKPOINT` to the downloaded public `model.safetensors`,
`TOMATOPGFM_TOKENIZER`,
`TOMATOPGFM_DATA_ROOT`, `DNABERT2_MODEL`, and `PLANTDNAMAMBA_MODEL` before
running `scripts/lora_genecds.py` or `scripts/benchmark_efficiency.py`. The
baseline repositories used were `zhihan1996/DNABERT-2-117M` and
`zhangtaolab/plant-dnamamba-BPE`; their exact downloaded revisions were not
recorded in the result files and must be fixed explicitly for a new rerun. The
efficiency script compares graph-off with `graph_mode="on"` supplied zero-valued
graph features and `edge_index=None`; this enables the GraphAdapter software
path without executing adjacency-based GraphMessage aggregation.

## Results

Machine-readable manuscript results are in `results/`. The final frozen-probe
JSON files and sample-level predictions correspond to the padding-aware
protocol. `results/graph_sensitivity/per_batch/` contains the ten paired inputs
required to rerun the reported bootstrap analysis.
