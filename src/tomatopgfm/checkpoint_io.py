"""Load inference weights and training checkpoints into a TomatoPGFM model."""

from __future__ import annotations

from pathlib import Path


CANONICAL_TIED_KEY = "embed.weight"
OMITTED_TIED_ALIAS = "mlm_head.weight"


def load_state_dict(path: str | Path):
    """Return a strict-loadable state dict from safetensors or a torch checkpoint."""
    import torch

    artifact = Path(path)
    if artifact.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(artifact), device="cpu")
        if CANONICAL_TIED_KEY in state and OMITTED_TIED_ALIAS not in state:
            state[OMITTED_TIED_ALIAS] = state[CANONICAL_TIED_KEY]
        return state

    try:
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions before the weights_only argument
        payload = torch.load(artifact, map_location="cpu")
    state = payload.get("model", payload.get("model_state_dict", payload))
    if not isinstance(state, dict):
        raise TypeError(f"No model state dictionary found in {artifact}")
    return state


def load_model_weights(model, path: str | Path):
    """Load either the public inference artifact or a training checkpoint strictly."""
    model.load_state_dict(load_state_dict(path), strict=True)
    return model
