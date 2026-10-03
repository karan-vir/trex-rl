"""The reward model: learns "how good is this moment?" from human comparisons of clips.

r(obs, action) -> one number. A clip's score is the average over its steps. Given two clips A and B, the model
says P(A preferred) = sigmoid(score(A) - score(B)) (Bradley-Terry); training maximises the likelihood of the
human verdicts. This is the same objective RLHF uses for LLM reward models, with clips instead of responses.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from trex.rlhf.core import T_SEG, load_prefs, replay

IN_DIM = 25                                    # 22 observation numbers + one-hot of the action taken


class RewardNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(IN_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sd", torch.ones(1))

    def forward(self, x):                      # raw per-step reward
        return self.net(x).squeeze(-1)

    def normalised(self, x):                   # zero mean, unit spread over the training clips
        return (self.forward(x) - self.mu) / self.sd


def step_rewards(rm: RewardNet, X: np.ndarray) -> list[float]:
    with torch.no_grad():
        return [round(float(v), 3) for v in rm.normalised(torch.as_tensor(X))]


def save_rm(rm: RewardNet, path, meta: dict) -> None:
    torch.save({"state": rm.state_dict(), "meta": meta}, path)


def load_rm(path) -> tuple[RewardNet, dict]:
    d = torch.load(path, map_location="cpu")
    rm = RewardNet()
    rm.load_state_dict(d["state"])
    rm.eval()
    return rm, d["meta"]


def _label_value(label: str) -> float:
    return {"A": 1.0, "B": 0.0, "same": 0.5}[label]


def featurise(records: list[dict]):
    """Replay every clip once -> padded tensors (N, T, 25), masks (N, T), targets (N)."""
    N = len(records)
    XA, XB = np.zeros((N, T_SEG, IN_DIM), np.float32), np.zeros((N, T_SEG, IN_DIM), np.float32)
    MA, MB = np.zeros((N, T_SEG), np.float32), np.zeros((N, T_SEG), np.float32)
    for i, r in enumerate(records):
        for X, M, key in ((XA, MA, "A"), (XB, MB, "B")):
            x = replay(r["spec"], r[key])["X"]
            X[i, :len(x)], M[i, :len(x)] = x, 1.0
    y = np.array([_label_value(r["label"]) for r in records], np.float32)
    return [torch.as_tensor(a) for a in (XA, XB, MA, MB, y)]


def pair_logit(rm, XA, XB, MA, MB):
    # Mean per step, not the sum: with a sum, "longer clip = more reward" would let the model win by
    # outputting a positive constant (a length bias, also a classic problem in LLM reward models).
    return (rm(XA) * MA).sum(1) / MA.sum(1).clamp_min(1) - (rm(XB) * MB).sum(1) / MB.sum(1).clamp_min(1)


def train_reward_model(records: list[dict], epochs: int = 60, lr: float = 3e-3, weight_decay: float = 1e-3, val_frac: float = 0.2,
                       seed: int = 0, progress=None):
    """Returns (model, history). history rows: epoch, train loss, train accuracy, validation accuracy."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    use = [r for r in records if r["label"] in ("A", "B", "same")]
    idx = rng.permutation(len(use))
    n_val = max(2, int(len(use) * val_frac)) if len(use) >= 10 else 0
    val_i, tr_i = idx[:n_val], idx[n_val:]
    XA, XB, MA, MB, y = featurise(use)
    rm = RewardNet()
    opt = torch.optim.Adam(rm.parameters(), lr=lr, weight_decay=weight_decay)
    hist = []

    def acc(ii):
        if len(ii) == 0:
            return None
        with torch.no_grad():
            lg = pair_logit(rm, XA[ii], XB[ii], MA[ii], MB[ii])
        decided = (y[ii] != 0.5)
        if decided.sum() == 0:
            return None
        return float((((lg > 0).float() == y[ii]) & decided).sum() / decided.sum())

    for ep in range(epochs):
        perm = rng.permutation(tr_i)
        tot = 0.0
        for b in range(0, len(perm), 64):
            ii = perm[b:b + 64]
            lg = pair_logit(rm, XA[ii], XB[ii], MA[ii], MB[ii])
            loss = nn.functional.binary_cross_entropy_with_logits(lg, y[ii])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(ii)
        hist.append(dict(epoch=ep + 1, loss=round(tot / max(1, len(perm)), 4), train_acc=acc(tr_i), val_acc=acc(val_i)))
        if progress:
            progress(ep + 1, epochs, hist[-1])
    with torch.no_grad():                       # normalise so a typical step has reward ~N(0, 1)
        allr = torch.cat([rm(XA[i])[MA[i] > 0] for i in tr_i[:200]] + [rm(XB[i])[MB[i] > 0] for i in tr_i[:200]])
        rm.mu[:] = allr.mean(); rm.sd[:] = allr.std().clamp_min(1e-3)
    rm.eval()
    return rm, hist, {"n_train": len(tr_i), "n_val": len(val_i)}


def accuracy_vs_judge(rm, model, n: int = 150, seed: int = 777) -> float:
    """On fresh clip pairs the RM never saw: how often does it agree with the hidden judge (decided pairs only)?"""
    from trex.rlhf.core import make_pair
    rng = np.random.default_rng(seed)
    ok = tot = 0
    while tot < n:
        p = make_pair(model, rng)
        if p["truth"] == "same":
            continue
        ra, rb = replay(p["spec"], p["A"]), replay(p["spec"], p["B"])
        with torch.no_grad():
            sa = float(rm(torch.as_tensor(ra["X"])).mean()); sb = float(rm(torch.as_tensor(rb["X"])).mean())
        ok += (sa > sb) == (p["truth"] == "A"); tot += 1
    return ok / tot
