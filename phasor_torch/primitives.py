"""Math primitives — phase/complex conversions, similarity, normalization.

Phase tensors are float32 in [-1, 1] (units of pi). Complex tensors are
torch.complex64. All shapes follow the Julia conventions documented in
the corresponding src/domains.jl, src/activations.jl, src/vsa.jl.

Two autograd.Function shims (`_NormalizeHard`, `_ComplexToAngle`) carry the
NaN guards that the Julia ChainRules versions have, because PyTorch's
native autograd for `torch.angle` and division-by-magnitude produces NaN
at z = 0 and poisons sibling cotangents via 0*NaN = NaN.

One closed-form autograd.Function (`_SimilarityOuterCanonicalComplex`)
mirrors the Julia memory optimization in src/vsa.jl:561 — avoids
materializing a (D, M, N, X) intermediate, which would blow up on long
sequences regardless of which AD framework you use.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

PI: float = math.pi


# --------------------------------------------------------------------------
# Phase <-> Complex
# --------------------------------------------------------------------------


def angle_to_complex(x: Tensor) -> Tensor:
    """Map phase tensor (real, [-1, 1] units of pi) to unit-circle complex.

    z = exp(i*pi*x). Mirrors Julia src/domains.jl:13.
    """
    return torch.exp(torch.complex(torch.zeros_like(x), PI * x))


class _ComplexToAngle(torch.autograd.Function):
    """Backward-safe angle(z)/pi with sub-threshold cotangent zeroed.

    Forward: y = atan2(imag(z), real(z)) / pi
    Backward (away from z = 0):  dz = ybar * i*z / (pi * |z|^2)
    At |z| <= threshold: forward returns 0, backward returns 0
    (matches Julia ChainRules rule in src/domains.jl:65).
    """

    @staticmethod
    def forward(ctx, z: Tensor, threshold: float) -> Tensor:
        zr = z.real
        zi = z.imag
        r2 = zr * zr + zi * zi
        th2 = float(threshold) ** 2
        active = r2 > th2
        y = torch.atan2(zi, zr) / PI
        y = torch.where(active, y, torch.zeros_like(y))
        ctx.save_for_backward(z, r2, active)
        ctx.th2 = th2
        return y

    @staticmethod
    def backward(ctx, ybar: Tensor):
        z, r2, active = ctx.saved_tensors
        safe_r2 = torch.clamp(r2, min=ctx.th2)
        # dz = ybar * (i * z) / (pi * |z|^2)
        dz_real = -ybar * z.imag / (PI * safe_r2)
        dz_imag = ybar * z.real / (PI * safe_r2)
        zero = torch.zeros_like(dz_real)
        dz_real = torch.where(active, dz_real, zero)
        dz_imag = torch.where(active, dz_imag, zero)
        dz = torch.complex(dz_real, dz_imag)
        return dz, None  # no grad for threshold


def complex_to_angle(z: Tensor, threshold: float = 1e-10) -> Tensor:
    """Map complex tensor to phase tensor (real, [-1, 1] units of pi).

    y = angle(z) / pi, with sub-threshold elements zeroed in both forward
    and backward to avoid NaN propagation from atan2(0, 0).
    """
    return _ComplexToAngle.apply(z, threshold)


# --------------------------------------------------------------------------
# Phase wrap and similarity
# --------------------------------------------------------------------------


def remap_phase(x: Tensor) -> Tensor:
    """Wrap phase tensor into [-1, 1] via mod-2 shift, no gradient.

    Mirrors Julia src/vsa.jl:309. Wrapped under no_grad because Julia
    wraps with ignore_derivatives.
    """
    with torch.no_grad():
        y = torch.remainder(x + 1.0, 2.0) - 1.0
    # detach to ensure the wrap is purely a value-space operation; gradients
    # of the unwrapped phase flow through the identity x -> x branch instead.
    if x.requires_grad:
        # straight-through: forward returns wrapped value, backward returns identity
        return x + (y - x).detach()
    return y


def similarity(x: Tensor, y: Tensor, dim: int = 0) -> Tensor:
    """Cosine of phase difference, averaged along `dim`.

    s = mean(cos(pi * (x - y)), dim). Mirrors Julia src/vsa.jl:334
    (note: Julia defaults to dim=1 which is the first axis; we keep
    a Python-flavored default of dim=0 for symmetry).
    """
    if dim < 0:
        dim = x.ndim + dim
    diff = torch.cos(PI * (x - y))
    return diff.mean(dim=dim)


# --------------------------------------------------------------------------
# Unit-circle normalization (smooth and hard)
# --------------------------------------------------------------------------


def _normalize_safe(z: Tensor, eps: float) -> Tensor:
    """Safe branch: y = z / sqrt(|z|^2 + eps). Smooth everywhere."""
    denom = torch.sqrt(z.real * z.real + z.imag * z.imag + float(eps))
    return z / denom.to(z.dtype)  # broadcast real denom over complex z


class _NormalizeHard(torch.autograd.Function):
    """Hard branch: y = z / |z| with `1 + 0i` fallback below threshold.

    Closed-form pullback returns zero cotangent for sub-threshold
    elements, matching Julia src/activations.jl:121. Forward outputs
    1 + 0i for elements where |z| <= threshold.
    """

    @staticmethod
    def forward(ctx, z: Tensor, threshold: float) -> Tensor:
        zr = z.real
        zi = z.imag
        r = torch.sqrt(zr * zr + zi * zi)
        th = float(threshold)
        safe_r = torch.clamp(r, min=th)
        unit_z = z / safe_r.to(z.dtype)
        default = torch.complex(torch.ones_like(zr), torch.zeros_like(zr))
        active = r > th
        y = torch.where(active, unit_z, default)
        ctx.save_for_backward(z, safe_r, active)
        return y

    @staticmethod
    def backward(ctx, ybar: Tensor):
        z, safe_r, active = ctx.saved_tensors
        # dz = -i * z * imag(z * conj(ybar)) / r^3
        zybar = z * ybar.conj()
        scale = zybar.imag / (safe_r ** 3)
        # multiply z by -i: (-i)(a+ib) = b - i*a
        dz_real = z.imag * scale
        dz_imag = -z.real * scale
        zero = torch.zeros_like(dz_real)
        dz_real = torch.where(active, dz_real, zero)
        dz_imag = torch.where(active, dz_imag, zero)
        dz = torch.complex(dz_real, dz_imag)
        return dz, None


def normalize_to_unit_circle(z: Tensor, eps: float = 1e-8,
                             threshold: float = 1e-10) -> Tensor:
    """Project complex tensor onto (or near) the unit circle.

    Two modes, selected by `eps`:
      - eps > 0 (default, safe): y = z / sqrt(|z|^2 + eps). Smooth
        everywhere; native autograd suffices.
      - eps == 0 (hard): y = z / |z| with `1 + 0i` fallback for
        |z| <= threshold; backward returns zero cotangent for those
        elements via _NormalizeHard.

    Mirrors Julia src/activations.jl:89.
    """
    if float(eps) == 0.0:
        return _NormalizeHard.apply(z, threshold)
    return _normalize_safe(z, eps)


# --------------------------------------------------------------------------
# Pairwise interference similarity (memory-efficient closed-form rrule)
# --------------------------------------------------------------------------


class _SimilarityOuterCanonicalComplex(torch.autograd.Function):
    """Closed-form pairwise interference similarity over a feature dim D.

    Forward and backward implement the math from Julia
    src/vsa.jl:561, avoiding any (D, M, N, X) intermediate. Both
    inputs are complex; output is real.

    Inputs:
      A: (D, M, X) complex
      B: (D, N, X) complex
    Output:
      s: (M, N, X) real, s[m,n,x] = (1/D) * sum_d (|A+B|^2 / 2 - 1)

    Equivalent expansion used by both forward and backward:
      s[m,n,x] = (1/D) [ 0.5 * a2[m,x] + 0.5 * b2[n,x] + cross[m,n,x] ] - 1
        a2[m,x]    = sum_d |A[d,m,x]|^2
        b2[n,x]    = sum_d |B[d,n,x]|^2
        cross[m,n,x] = sum_d (Ar*Br + Ai*Bi)
    """

    @staticmethod
    def forward(ctx, A: Tensor, B: Tensor) -> Tensor:
        D, M, X = A.shape
        _, N, _ = B.shape
        inv_D = 1.0 / float(D)

        Ar, Ai = A.real, A.imag       # (D, M, X)
        Br, Bi = B.real, B.imag       # (D, N, X)

        a2 = (Ar * Ar + Ai * Ai).sum(dim=0, keepdim=False).unsqueeze(1)   # (M, 1, X)
        b2 = (Br * Br + Bi * Bi).sum(dim=0, keepdim=False).unsqueeze(0)   # (1, N, X)

        # cross = batched_mul(Ar^T, Br) + batched_mul(Ai^T, Bi)
        # Layout: bmm wants (batch, M, K) @ (batch, K, N) -> (batch, M, N).
        # Our batch dim is X, K = D. So permute to (X, M, D) and (X, D, N).
        Ar_b = Ar.permute(2, 1, 0).contiguous()   # (X, M, D)
        Ai_b = Ai.permute(2, 1, 0).contiguous()   # (X, M, D)
        Br_b = Br.permute(2, 0, 1).contiguous()   # (X, D, N)
        Bi_b = Bi.permute(2, 0, 1).contiguous()   # (X, D, N)
        cross_xMN = torch.bmm(Ar_b, Br_b) + torch.bmm(Ai_b, Bi_b)  # (X, M, N)
        cross = cross_xMN.permute(1, 2, 0).contiguous()             # (M, N, X)

        out = inv_D * (0.5 * (a2 + b2) + cross) - 1.0
        ctx.save_for_backward(A, B)
        ctx.inv_D = inv_D
        return out

    @staticmethod
    def backward(ctx, gbar: Tensor):
        A, B = ctx.saved_tensors
        inv_D = ctx.inv_D
        D, M, X = A.shape
        _, N, _ = B.shape

        # g_row[m, x] = sum_n gbar[m, n, x],  shape (1, M, X) for broadcast over D
        g_row = gbar.sum(dim=1, keepdim=True).permute(2, 1, 0)   # (X, 1, M) -> reshaped below
        # actually compute as (1, M, X)
        g_row_dmX = gbar.sum(dim=1).unsqueeze(0)   # (1, M, X)
        g_col_dnX = gbar.sum(dim=0).unsqueeze(0)   # (1, N, X)

        # AB_term[d, m, x] = sum_n B[d, n, x] * gbar[m, n, x]
        # batched on X: B_b = (X, D, N); gbar_b = (X, N, M); result = (X, D, M)
        gbar_cpx = gbar.to(A.dtype)
        B_b = B.permute(2, 0, 1).contiguous()                              # (X, D, N)
        gT_b = gbar_cpx.permute(2, 1, 0).contiguous()                      # (X, N, M)
        AB_xDM = torch.bmm(B_b, gT_b)                                      # (X, D, M)
        AB_term = AB_xDM.permute(1, 2, 0).contiguous()                     # (D, M, X)

        # BA_term[d, n, x] = sum_m A[d, m, x] * gbar[m, n, x]
        A_b = A.permute(2, 0, 1).contiguous()                              # (X, D, M)
        g_b = gbar_cpx.permute(2, 0, 1).contiguous()                       # (X, M, N)
        BA_xDN = torch.bmm(A_b, g_b)                                       # (X, D, N)
        BA_term = BA_xDN.permute(1, 2, 0).contiguous()                     # (D, N, X)

        dA = inv_D * (A * g_row_dmX + AB_term)
        dB = inv_D * (B * g_col_dnX + BA_term)
        return dA, dB


def _similarity_outer_canonical_complex(A: Tensor, B: Tensor) -> Tensor:
    """Public-ish entry: shape-checks and routes through the autograd.Function."""
    assert A.is_complex() and B.is_complex(), "complex inputs required"
    assert A.ndim == 3 and B.ndim == 3, "rank-3 (D, M/N, X) inputs required"
    assert A.shape[0] == B.shape[0], f"feature dim mismatch {A.shape[0]} vs {B.shape[0]}"
    assert A.shape[2] == B.shape[2], f"batch dim mismatch {A.shape[2]} vs {B.shape[2]}"
    return _SimilarityOuterCanonicalComplex.apply(A, B)


def similarity_outer_heads(q: Tensor, k: Tensor) -> Tensor:
    """Head-axis pairwise similarity for LSA/LCA.

    Two shape regimes (matched on caller-side; this dispatches on ndim):

      LSA  q: (Dh, H, L, B) phase  k: (Dh, H, L, B) phase  ->  (H, H, L, B)
      LCA  q: (Dh, H, A)    phase  k: (Dh, H, L, B) phase  ->  (A, H, L, B)

    Mirrors Julia src/vsa.jl:657 (LSA) and src/vsa.jl:672 (LCA).
    """
    if q.ndim == 4 and k.ndim == 4:
        Dh, H, L, B = q.shape
        assert k.shape == q.shape, f"LSA shape mismatch q={q.shape} k={k.shape}"
        qc = angle_to_complex(q)                 # (Dh, H, L, B) complex
        kc = angle_to_complex(k)
        Aq = qc.reshape(Dh, H, L * B)
        Bk = kc.reshape(Dh, H, L * B)
        s = _similarity_outer_canonical_complex(Aq, Bk)  # (H, H, L*B)
        return s.reshape(H, H, L, B)

    if q.ndim == 3 and k.ndim == 4:
        Dh, H, A = q.shape
        Dh_k, H_k, L, B = k.shape
        assert Dh_k == Dh and H_k == H, (
            f"LCA channel/head mismatch q={q.shape} k={k.shape}"
        )
        qc = angle_to_complex(q)                  # (Dh, H, A)
        kc = angle_to_complex(k)                  # (Dh, H, L, B)
        per_head = []
        for h in range(H):
            Aq_h = qc[:, h, :].reshape(Dh, A, 1)             # (Dh, A, 1)
            Bk_h = kc[:, h, :, :].reshape(Dh, L * B, 1)      # (Dh, L*B, 1)
            s_h = _similarity_outer_canonical_complex(Aq_h, Bk_h)  # (A, L*B, 1)
            per_head.append(s_h.reshape(A, L, B))
        stacked = torch.stack(per_head, dim=-1)              # (A, L, B, H)
        return stacked.permute(0, 3, 1, 2).contiguous()      # (A, H, L, B)

    raise ValueError(
        f"similarity_outer_heads: unsupported shapes q.ndim={q.ndim}, k.ndim={k.ndim}"
    )
