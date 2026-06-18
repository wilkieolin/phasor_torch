"""HDF5 weight serialization, schema-compatible with Lux Chain.

Schema:
  /<layer_path>/<param_name>        float32 dataset

For a Lux Chain like
  Chain(input=PhasorDense(in=>D), attn=PhasorLSA(D=>D), body=PhasorDense(D=>D), readout=SSMReadout(D=>K))
the file contains
  /input/weight
  /input/log_neg_lambda
  /input/bias_real        (if use_bias)
  /input/bias_imag        (if use_bias)
  /attn/q_proj/weight
  /attn/q_proj/log_neg_lambda
  /attn/k_proj/weight
  ...
  /attn/scale
  /body/weight
  /readout/codes

Buffers and shared per-layer constants (e.g. omega) are NOT serialized —
both sides reconstruct them from each layer's constructor config.

The PyTorch side serializes via `save_state(model, path, schema)` and
loads via `load_state(model, path, schema)`, where `schema` is a dict
mapping HDF5 group paths to nn.Module instances. The matching Julia
helpers in julia_parity/load_pytorch.jl traverse the same paths into a
Lux NamedTuple.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np
import torch
from torch import Tensor, nn


def _to_numpy(t: Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _save_module(group: h5py.Group, module: nn.Module) -> None:
    """Write a module's `parameter_dict()` into an HDF5 group.

    The module is expected to expose a `parameter_dict()` returning
    {param_name: Tensor}. Falls back to `state_dict()` if not present
    (best-effort).
    """
    if hasattr(module, "parameter_dict"):
        params = module.parameter_dict()
    else:
        params = {k: v for k, v in module.state_dict().items()}
    for name, tensor in params.items():
        arr = _to_numpy(tensor)
        # Overwrite if it already exists (idempotent re-save).
        if name in group:
            del group[name]
        group.create_dataset(name, data=arr)


def _resolve_param(module: nn.Module, path: str):
    """Walk a slash-separated path of attributes to a Parameter or buffer.

    Supports nested layers (e.g. 'q_proj/weight'): descends through
    intermediate nn.Modules via getattr, then resolves the final segment
    on the leaf. Returns the resolved tensor reference (Parameter or
    buffer) and the parent module.
    """
    parts = path.split("/")
    target = module
    for p in parts[:-1]:
        if not hasattr(target, p):
            raise AttributeError(
                f"module {type(module).__name__} has no submodule '{p}' in path '{path}'"
            )
        target = getattr(target, p)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise AttributeError(
            f"module {type(target).__name__} has no attribute '{leaf}' (path '{path}')"
        )
    tensor = getattr(target, leaf)
    if tensor is None:
        raise AttributeError(
            f"path '{path}' resolves to None on {type(target).__name__}"
        )
    return tensor


def _load_module(group: h5py.Group, module: nn.Module) -> None:
    """Load a module's parameters from an HDF5 group, in-place.

    Nested parameter paths (e.g. 'q_proj/weight') become nested HDF5
    groups and are walked through module attributes.
    """
    target_keys = (
        list(module.parameter_dict().keys())
        if hasattr(module, "parameter_dict")
        else list(dict(module.state_dict()).keys())
    )
    for name in target_keys:
        if name not in group:
            raise KeyError(f"missing dataset '{name}' under {group.name}")
        arr = np.asarray(group[name])
        tensor = torch.from_numpy(arr).to(torch.float32)
        target = _resolve_param(module, name)
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(
                f"shape mismatch loading {group.name}/{name}: "
                f"file {tuple(tensor.shape)} vs module {tuple(target.shape)}"
            )
        with torch.no_grad():
            target.copy_(tensor)


def save_state(path: str | Path, schema: Mapping[str, nn.Module],
               metadata: Mapping[str, str] | None = None) -> None:
    """Save a {layer_path: module} schema to HDF5.

    Args:
      path:     output filename (.h5).
      schema:   ordered mapping; keys become HDF5 group paths (slash-separated
                allowed for nested layers, e.g. 'attn/q_proj').
      metadata: optional string attributes attached to the root group.
    """
    path = Path(path)
    with h5py.File(path, "w") as f:
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = str(v)
        for group_path, module in schema.items():
            grp = f.require_group(group_path)
            _save_module(grp, module)


def load_state(path: str | Path, schema: Mapping[str, nn.Module]) -> None:
    """Load weights from an HDF5 file into the modules of `schema`."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        for group_path, module in schema.items():
            if group_path not in f:
                raise KeyError(f"missing group '{group_path}' in {path}")
            _load_module(f[group_path], module)


def save_io_pair(
    path: str | Path,
    inputs: Mapping[str, Tensor],
    outputs: Mapping[str, Tensor],
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Save a (canned input, computed output) pair for parity verification.

    The Julia parity script loads this, re-runs the equivalent forward on
    the corresponding HDF5 weight file, and asserts agreement.

    Layout:
      /inputs/<name>     float32 (real)  or  /inputs/<name>_real + _imag for complex
      /outputs/<name>    same encoding
    """
    path = Path(path)
    with h5py.File(path, "w") as f:
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = str(v)
        for label, group_dict in (("inputs", inputs), ("outputs", outputs)):
            grp = f.require_group(label)
            for name, tensor in group_dict.items():
                arr = tensor.detach().cpu().numpy()
                if np.iscomplexobj(arr):
                    grp.create_dataset(f"{name}_real", data=arr.real.astype("float32"))
                    grp.create_dataset(f"{name}_imag", data=arr.imag.astype("float32"))
                else:
                    grp.create_dataset(name, data=arr.astype("float32"))
