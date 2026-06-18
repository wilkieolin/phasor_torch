"""PyTorch port of PhasorNetworks.jl — training-only, LSA/LCA focused.

Targets Intel PVC (xpu) on Aurora; works on any PyTorch backend that supports
torch.complex64 and torch.fft.

Layout:
  primitives.py    — angle_to_complex, complex_to_angle, normalize_to_unit_circle,
                     remap_phase, similarity, similarity_outer_heads
  kernels.py       — phasor_kernel, causal_conv (Toeplitz/FFT), causal_conv_dirac,
                     hippo_legs_diagonal
  layers/          — PhasorDense, PhasorLSA, PhasorLCA, Codebook, SSMReadout
  losses.py        — similarity_loss, codebook_loss, accuracy
  weights.py       — state_dict <-> HDF5 (weight-compat with Julia Lux Chain)
  config.py        — model + training dataclasses
  train.py         — Adam loop with logging and checkpointing
  data/            — sequence task generators (copy, reversal, retrieval, …)

Phase tensors are plain real-valued torch.Tensor in [-1, 1] (units of pi).
There is no Phase wrapper class — the [-1, 1] invariant is documented
per-function rather than dispatch-enforced.
"""

from .primitives import (
    PI,
    angle_to_complex,
    complex_to_angle,
    normalize_to_unit_circle,
    remap_phase,
    similarity,
    similarity_outer_heads,
)
from .kernels import (
    causal_conv,
    causal_conv_dirac,
    hippo_legs_diagonal,
    phasor_kernel,
)

__all__ = [
    "PI",
    "angle_to_complex",
    "causal_conv",
    "causal_conv_dirac",
    "complex_to_angle",
    "hippo_legs_diagonal",
    "normalize_to_unit_circle",
    "phasor_kernel",
    "remap_phase",
    "similarity",
    "similarity_outer_heads",
]
