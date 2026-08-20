# TomatoPGFM model card

## Model

TomatoPGFM is a 479,195,678-parameter graph-conditioned tomato genomic foundation model. Approximately 148,625,438 parameters are active per token under top-2 expert routing. The model consumes reverse-complement-folded non-overlapping 6-mers, an eight-channel token-aligned graph interface and optional within-window adjacency. In the production shards, five graph-derived channels vary and three interface channels are constant zero.

## Training data

The pretraining panel contains 66 tomato accessions and 54.65 Gb of assembled sequence. LA1974 and MicroTom were excluded from graph construction and pretraining.

## Intended use

The checkpoint supports research on tomato genomic representation learning, graph-conditioned masked-token prediction and downstream sequence classification. Users must independently validate predictions before biological or breeding decisions.

## Evaluation

The repository includes graph-input sensitivity, frozen gene/CDS probes, LoRA adaptation and zero-feature GraphAdapter software-path efficiency results corresponding to the manuscript.

## Distribution

The primary public artifact is an inference-only safetensors state dictionary.
The optimizer-bearing training-resume checkpoint is not included in the public
release; its identity is retained only as export provenance. Public archive URLs
and the permanent DOI are added to the living release metadata after publication.
