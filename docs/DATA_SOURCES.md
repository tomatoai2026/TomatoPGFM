# Data sources

The 66-accession training panel comprised 39 accessions from Zhou et al. (2022; NCBI BioProject PRJNA733299), eight from Li et al. (2023; NCBI BioProject PRJNA809001), and 19 from Shi et al. (2026; CNCB PRJCA030093, NCBI PRJNA1201608, Zenodo 10.5281/zenodo.17878268) after LA1974 was reserved for external evaluation. LA1974 (NCBI PRJNA633104) and MicroTom (RefSeq GCF_036512215.1) were used only for downstream evaluation.

Large assemblies, annotations, graph files, training shards and baseline model
weights are not redistributed in this GitHub package. The public training-panel
table retains accession identifiers, source collections, counts and source-file
checksums but removes local filesystem paths and timestamps.

The evaluation manifests distribute exact 0-based half-open coordinates,
labels, row order and per-window sequence SHA-256 values. They do not
redistribute the 512-bp sequence strings. Download the cited LA1974 or MicroTom
assembly and run `scripts/materialize_manifest_sequences.py` to reconstruct and
checksum-verify sequence-bearing manifests.
