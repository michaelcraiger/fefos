"""A compact Roost-style network (Goodall & Lee, Nat. Commun. 11, 6280, 2020).

Re-implemented here so the benchmark needs no third-party model code. As in
Roost, each compound is a dense graph of its elements; node states are updated
by message passing with fraction-weighted soft attention, and a fraction-
weighted attention pool gives the compound representation. Element embeddings
are learned from scratch (Roost used pre-trained Matscholar embeddings) and the
model is smaller and trained for fewer epochs, so this is a "Roost-style"
baseline, not the published model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

Z = {
    s: i + 1
    for i, s in enumerate(
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge "
        "As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm "
        "Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U "
        "Np Pu".split()
    )
}
MAX_EL = 9


def encode(compositions: list[dict[str, float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Element indices and fractions, padded to ``MAX_EL`` (index 0 = padding)."""
    idx = np.zeros((len(compositions), MAX_EL), dtype=np.int64)
    frac = np.zeros((len(compositions), MAX_EL), dtype=np.float32)
    for n, comp in enumerate(compositions):
        total = sum(comp.values())
        for k, (el, x) in enumerate(sorted(comp.items())):
            idx[n, k] = Z[el]
            frac[n, k] = x / total
    return torch.from_numpy(idx), torch.from_numpy(frac)


class _Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.message = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, h, frac, mask):
        n = h.shape[1]
        pair = torch.cat(
            [h.unsqueeze(2).expand(-1, -1, n, -1), h.unsqueeze(1).expand(-1, n, -1, -1)], -1
        )
        logits = self.gate(pair).squeeze(-1)  # B x n x n
        weight = frac.unsqueeze(1) * torch.exp(logits - logits.amax(-1, keepdim=True))
        weight = weight * mask.unsqueeze(1)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
        return h + torch.einsum("bij,bijd->bid", weight, self.message(pair))


class RoostLite(nn.Module):
    def __init__(self, dim: int = 64, layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(len(Z) + 1, dim, padding_idx=0)
        self.layers = nn.ModuleList(_Attention(dim) for _ in range(layers))
        self.pool_gate = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.head = nn.Sequential(
            nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def forward(self, idx, frac):
        mask = (idx > 0).float()
        h = self.embed(idx)
        for layer in self.layers:
            h = layer(h, frac, mask)
        g = self.pool_gate(h).squeeze(-1)
        w = frac * torch.exp(g - g.amax(-1, keepdim=True)) * mask
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
        return self.head(torch.einsum("bi,bid->bd", w, h)).squeeze(-1)


def train_predict(
    train_comps, train_y, test_comps, epochs: int = 60, seed: int = 0, device: str | None = None
) -> np.ndarray:
    """Train on ``train_comps`` (list of element->amount dicts) and predict ``test_comps``."""
    torch.manual_seed(seed)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    idx, frac = encode(train_comps)
    y = torch.tensor(np.asarray(train_y, dtype=np.float32))
    model = RoostLite().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=epochs * max(1, int(np.ceil(len(y) / 256)))
    )
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=gen)
        for start in range(0, len(y), 256):
            b = perm[start : start + 256]
            pred = model(idx[b].to(device), frac[b].to(device))
            loss = torch.nn.functional.l1_loss(pred, y[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    tidx, tfrac = encode(test_comps)
    out = []
    with torch.no_grad():
        for start in range(0, len(test_comps), 1024):
            out.append(
                model(
                    tidx[start : start + 1024].to(device), tfrac[start : start + 1024].to(device)
                ).cpu()
            )
    return torch.cat(out).numpy()
