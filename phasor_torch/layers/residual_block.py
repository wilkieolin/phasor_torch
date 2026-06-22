"""PhasorResidualBlock — a (body -> dense) block with VSA-binding residuals.

Phase-domain residual block, mirroring Julia SingleHeadCABlock
(PhasorNetworks.jl src/network.jl:1797): a skip is a `v_bind` (phase addition
with wrap), not an additive `x + f(x)`. Two binds per block — one around the
attention body, one around the dense/FFN:

    h   = v_bind(x, body(x))      # residual around attention (skipped if body is None)
    out = v_bind(h, dense(h))     # residual around the FFN

Because `v_bind == remap_phase(x + y)` is straight-through, each bind contributes
an identity term to the gradient, giving deep stacks a vanishing/exploding-proof
gradient highway. Input/output are Phase `(d_hidden, L, B)` (same shape).
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from ..primitives import v_bind


class PhasorResidualBlock(nn.Module):
    """Wrap a body (PhasorLCA/LSA or None) + dense (PhasorDense) with v_bind skips."""

    def __init__(self, body: Optional[nn.Module], dense: nn.Module):
        super().__init__()
        self.body = body          # None for the 'none' baseline (dense-only block)
        self.dense = dense

    def forward(self, x: Tensor) -> Tensor:
        h = v_bind(x, self.body(x)) if self.body is not None else x
        return v_bind(h, self.dense(h))

    def parameter_dict(self) -> dict[str, Tensor]:
        """Flat slash-namespaced params: body/<...> and dense/<...>.

        Maps onto the Julia SingleHeadCABlock parameter tree (attn + ff) for
        HDF5 round trip / Lux loading.
        """
        out: dict[str, Tensor] = {}
        if self.body is not None:
            for k, v in self.body.parameter_dict().items():
                out[f"body/{k}"] = v
        for k, v in self.dense.parameter_dict().items():
            out[f"dense/{k}"] = v
        return out
