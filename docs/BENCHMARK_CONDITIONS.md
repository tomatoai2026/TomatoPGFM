# Efficiency benchmark conditions

The manuscript efficiency comparison uses two TomatoPGFM software paths:

- `adapter-on`: `graph_mode="on"`, zero-valued graph features, and `edge_index=None`.
- `graph-off`: `graph_mode="off"`, with both graph-conditioning pathways disabled.

Because `edge_index=None` in the adapter-on condition, adjacency-based GraphMessage aggregation is not executed. The historical result key `tomatopgfm_graph_on` is retained for file compatibility and maps to the `adapter-on` condition above. The numerical benchmark values are unchanged.
