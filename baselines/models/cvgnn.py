"""Complex-valued GNN baseline (Jin et al., "Efficient
Radiation Pattern Prediction of Array Antennas Based on Complex-Valued Graph
Neural Networks", IEEE AWPL 2022).

Forward predictor: the array is a graph (nodes = elements with complex excitation
feature [Re(w), Im(w)], edges = relative positions between elements within 0.5λ),
encoded by a 3-layer GIN backbone, then a complex-weighted FCN head with a
residual connection predicts the radiation pattern.

The array here is a fixed N×N half-wavelength lattice, so the graph is fixed and
only the node excitations vary with the steering (dpx, dpy). The model accepts
the same input tensor as the BO-CNN baseline, R = [x, y, Re(w), Im(w)] of shape
(B, 4, N), so both run through the same training and evaluation code.

Fidelity notes (vs paper):
- Node feature = [Re(w), Im(w)] (complex excitation), edges = relative positions,
  adjacency 0<d≤0.5λ → 4 orthogonal lattice neighbours (diagonals at 0.707λ are
  excluded, per the paper's Eq. 4). GIN message-passing per the paper's Eq. 5.
  The GIN backbone (the substantive difference from the BO-CNN's Euclidean conv
  backbone) uses intra-layer residuals to avoid oversmoothing, plus a skip
  connection from the raw node features to the head.
- head="complex" is the published head: complex-weighted FC layers with residual
  connections (Eq. 6-8), magnitude output, trained with L1. This is the variant
  reported in the paper comparison.
- head="mlp" is a simplified real MLP head with a Sigmoid output, kept as an
  alternative.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 4 orthogonal neighbours (d = 0.5λ).  Diagonals (0.707λ) excluded per Eq. 4.
_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class ComplexLinear(nn.Module):
    """Complex fully-connected layer (paper Eq. 6–7): W·z with W=P+iQ, z=p+iq."""

    def __init__(self, in_f: int, out_f: int, bias: bool = True):
        super().__init__()
        self.P = nn.Linear(in_f, out_f, bias=False)
        self.Q = nn.Linear(in_f, out_f, bias=False)
        if bias:
            self.b_re = nn.Parameter(torch.zeros(out_f))
            self.b_im = nn.Parameter(torch.zeros(out_f))
        else:
            self.register_parameter("b_re", None)
            self.register_parameter("b_im", None)

    def forward(self, re, im):
        out_re = self.P(re) - self.Q(im)
        out_im = self.P(im) + self.Q(re)
        if self.b_re is not None:
            out_re = out_re + self.b_re
            out_im = out_im + self.b_im
        return out_re, out_im


def _mlp(in_d, out_d):
    return nn.Sequential(nn.Linear(in_d, out_d), nn.ReLU(inplace=True),
                         nn.Linear(out_d, out_d))


class GINLayer(nn.Module):
    """GIN message-passing on a fixed lattice (paper Eq. 5).

    h_i = f_v( h_i , sum_{u in N(i)} f_e(h_i, h_u, e_iu) )   over 4 neighbours.
    Operates on a (B, n, n, d) grid via directional shifts (efficient, fixed graph).
    """

    def __init__(self, in_d: int, out_d: int, edge_d: int = 2):
        super().__init__()
        self.f_e = _mlp(2 * in_d + edge_d, out_d)
        self.f_v = nn.Sequential(nn.Linear(in_d + out_d, out_d), nn.ReLU(inplace=True))

    def forward(self, H, edge_vec):
        # H: (B, n, n, d);  edge_vec: list dir-> (2,) tensor
        B, n, _, d = H.shape
        agg = H.new_zeros(B, n, n, self.f_v[0].out_features)
        for k, (di, dj) in enumerate(_DIRS):
            nbr = torch.roll(H, shifts=(-di, -dj), dims=(1, 2))
            mask = H.new_ones(n, n)
            if di == 1:  mask[-1, :] = 0
            if di == -1: mask[0, :] = 0
            if dj == 1:  mask[:, -1] = 0
            if dj == -1: mask[:, 0] = 0
            e = edge_vec[k].view(1, 1, 1, -1).expand(B, n, n, -1)
            msg = self.f_e(torch.cat([H, nbr, e], dim=-1))
            agg = agg + msg * mask.view(1, n, n, 1)
        out = self.f_v(torch.cat([H, agg], dim=-1))
        if out.shape[-1] == H.shape[-1]:
            out = out + H                          # residual: prevents oversmoothing
        return out


class ComplexResidualFCN(nn.Module):
    """Paper-faithful head (Jin et al. Eq. 6-8): a stack of complex-weighted FC
    layers with CReLU activations and residual connections between equal-width
    layers, followed by a complex output layer; the predicted pattern is the
    magnitude |z| of the complex output (linear head, no Sigmoid)."""

    def __init__(self, in_f: int, hidden: int, out_f: int, n_layers: int = 4):
        super().__init__()
        self.inp = ComplexLinear(in_f, hidden)
        self.hid = nn.ModuleList(ComplexLinear(hidden, hidden) for _ in range(max(0, n_layers - 2)))
        self.out = ComplexLinear(hidden, out_f)

    @staticmethod
    def _crelu(re, im):
        return F.relu(re), F.relu(im)

    def forward(self, re, im):
        re, im = self._crelu(*self.inp(re, im))
        for layer in self.hid:
            r2, i2 = self._crelu(*layer(re, im))
            re, im = re + r2, im + i2                # residual connection (Eq. 8)
        re, im = self.out(re, im)
        return torch.sqrt(re * re + im * im + 1e-12)  # |z| -> real pattern


class CVGNNBaseline(nn.Module):
    """Complex-valued GNN forward predictor.

    head="mlp"     : simplified real MLP + Sigmoid
    head="complex" : paper-faithful complex-weighted residual FCN, |z| output
    """

    def __init__(self, n_elem: int, out_dim: int,
                 gin_hidden: int = 32, gin_layers: int = 3, fc_hidden: int = 256,
                 head: str = "mlp", head_layers: int = 4):
        super().__init__()
        self.head_kind = head
        self.n = int(round(math.sqrt(n_elem)))
        assert self.n * self.n == n_elem, "CVGNN expects a square lattice"
        self.out_dim = out_dim

        # GIN backbone: node feature [Re(w), Im(w)] (dim 2) -> gin_hidden
        dims = [2] + [gin_hidden] * gin_layers
        self.gins = nn.ModuleList(GINLayer(dims[i], dims[i + 1]) for i in range(gin_layers))

        # fixed edge features (relative position per direction, in 0.5λ units)
        for k, (di, dj) in enumerate(_DIRS):
            self.register_buffer(f"edge_{k}", torch.tensor([float(di), float(dj)]))

        # head input = GIN node embeddings ++ raw node features (skip connection so
        # the per-element excitation always reaches the head)
        flat = n_elem * gin_hidden + n_elem * 2
        if head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(flat, fc_hidden), nn.ReLU(inplace=True),
                nn.Linear(fc_hidden, out_dim), nn.Sigmoid(),
            )
        elif head == "complex":
            assert gin_hidden % 2 == 0, "complex head splits GIN channels into re/im halves"
            # complex node vector: GIN embedding split as (re | im) halves ++ raw excitation w
            cx_in = n_elem * (gin_hidden // 2) + n_elem
            self.head = ComplexResidualFCN(cx_in, fc_hidden, out_dim, n_layers=head_layers)
        else:
            raise ValueError(head)

    def _edges(self):
        return [getattr(self, f"edge_{k}") for k in range(len(_DIRS))]

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        # R: (B, 4, N) = [x, y, Re(w), Im(w)]; node feature = [Re(w), Im(w)]
        B = R.shape[0]
        H0 = R[:, 2:4, :].transpose(1, 2).reshape(B, self.n, self.n, 2).contiguous()
        edges = self._edges()
        H = H0
        for gin in self.gins:
            H = gin(H, edges)
        if self.head_kind == "mlp":
            return self.head(torch.cat([H.reshape(B, -1), H0.reshape(B, -1)], dim=-1))
        h = H.shape[-1] // 2
        re = torch.cat([H[..., :h].reshape(B, -1), H0[..., 0].reshape(B, -1)], dim=-1)
        im = torch.cat([H[..., h:].reshape(B, -1), H0[..., 1].reshape(B, -1)], dim=-1)
        return self.head(re, im)

    def set_output_bias(self, mean01: np.ndarray):
        """Init output to predict the per-pixel training mean (escape flat plateau)."""
        with torch.no_grad():
            if self.head_kind == "mlp":
                m = np.clip(mean01, 1e-3, 1 - 1e-3).astype(np.float32)
                final = self.head[-2]   # last Linear (head[-1] is Sigmoid)
                final.bias.copy_(torch.from_numpy(np.log(m / (1 - m))).to(final.bias.device))
                final.weight.mul_(0.1)
            else:
                final = self.head.out   # complex output layer: |b_re + i*0| = mean
                final.b_re.copy_(torch.from_numpy(mean01.astype(np.float32)).to(final.b_re.device))
                final.b_im.zero_()
                final.P.weight.mul_(0.1); final.Q.weight.mul_(0.1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
