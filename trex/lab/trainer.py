"""The training-setup menu of the lab: every knob, its allowed values, and what it does in plain words.

Kept free of pygame so it can be tested: `Setup` holds the current choice for each knob and
`Setup.command()` turns it into the `scripts/train_ppo.py` command line.
"""

from __future__ import annotations

import sys
import time

# key, label, values, default, flag, unit-formatter, explanation
KNOBS = [
    ("frames", "length of run", [1e6, 2e6, 3e6, 5e6, 10e6, 20e6, 50e6], 3e6, "--frames",
     lambda v: f"{v / 1e6:g}M decisions",
     "How long to train. One decision is ~1/60 s of game time. 1M takes roughly 4-8 minutes on this machine; the real gains took 10-20M."),
    ("lr", "learning rate", [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 1e-3], 5e-5, "--lr", lambda v: f"{v:g}",
     "Size of each weight update. Too high: the policy swings wildly between checkpoints (what we saw). Too low: learns slowly. Fine-tuning a good model wants small, 3e-4 is for fresh networks."),
    ("ent", "entropy bonus", [0.0, 0.001, 0.003, 0.01, 0.03], 0.003, "--ent-coef", lambda v: f"{v:g}",
     "Reward for staying a little random. More = explores more, learns new tricks, but plays less crisply. Lower it late in training."),
    ("clip", "PPO clip range", [0.05, 0.1, 0.2, 0.3], 0.1, "--clip", lambda v: f"{v:g}",
     "PPO's safety rail: how far one update may move the policy. Smaller = steadier but slower."),
    ("scen", "dangerous starts", [0.0, 0.2, 0.4, 0.6, 0.8], 0.4, "--scenario-prob", lambda v: f"{v:.0%} of games",
     "Share of training games that begin with close obstacles, overhead pterodactyls, tight pairs. Practice on the rare cases the agent dies to."),
    ("gamma", "discount (gamma)", [0.95, 0.99, 0.995, 0.999], 0.995, "--gamma", lambda v: f"{v:g}",
     "How much the agent cares about the future. 0.995 looks ~200 decisions (3 s) ahead. Higher = plans further, noisier learning."),
    ("lam", "GAE lambda", [0.9, 0.95, 0.98], 0.95, "--gae-lambda", lambda v: f"{v:g}",
     "Bias/variance trade-off when judging whether an action was good. Rarely worth changing."),
    ("epochs", "epochs per batch", [3, 6, 10], 6, "--epochs", lambda v: f"{v:g}",
     "How many times each batch of experience is reused. More = squeezes more from data, risks over-fitting it."),
    ("n_steps", "steps per update", [256, 512, 1024], 512, "--n-steps", lambda v: f"{v:g} x envs",
     "Experience collected per environment before each update."),
    ("batch", "minibatch size", [512, 1024, 2048], 1024, "--batch-size", lambda v: f"{v:g}",
     "Samples per gradient step. Bigger = smoother gradient."),
    ("envs", "parallel games", [4, 8, 12], 8, "--n-envs", lambda v: f"{v:g}",
     "Games simulated at once (one CPU process each). More = faster, until you run out of cores."),
    ("alive", "reward: alive", [0.0, 0.005, 0.01, 0.02], 0.01, "--alive-reward", lambda v: f"{v:g} / decision",
     "Small reward for every decision survived. Pushes toward long runs; too big and it plays safe/passive."),
    ("passb", "reward: pass obstacle", [0.0, 0.1, 0.3, 0.6, 1.0], 0.3, "--pass-bonus", lambda v: f"{v:g}",
     "Bonus for clearing an obstacle. This is what let it learn at all: it tells the agent which jumps were right."),
    ("death", "penalty: crash", [0.5, 1.0, 2.0, 5.0], 1.0, "--death-penalty", lambda v: f"{v:g}",
     "Cost of dying. Bigger = more cautious."),
    ("cap", "episode cap", [5_000, 20_000, 60_000], 20_000, "--max-frames", lambda v: f"{v:g} decisions",
     "Training games stop after this many decisions (20,000 = ~5.5 min of play). Evals use the same cap."),
    ("eval", "evaluate every", [250_000, 500_000, 1_000_000], 500_000, "--eval-every", lambda v: f"{v / 1e6:g}M",
     "How often to play fixed test games, write a point on the learning curve and save a checkpoint."),
    ("width", "network width", [64, 128, 256], 128, "--net-width", lambda v: f"{v:g} units",
     "Units per hidden layer. Only for a brand-new network (not when continuing a checkpoint)."),
    ("layers", "network depth", [1, 2, 3], 2, "--net-layers", lambda v: f"{v:g} hidden layers",
     "Hidden layers. Only for a brand-new network."),
    ("seed", "random seed", [0, 1, 2, 3, 4, 5], 0, "--seed", lambda v: f"{v:g}",
     "Different seeds give different training runs; handy to see how much luck matters."),
]
NEW_ONLY = {"width", "layers"}

PRESETS = {
    "polish a checkpoint": dict(frames=3e6, lr=5e-5, ent=0.003, clip=0.1, scen=0.4),
    "fresh network (original recipe)": dict(frames=20e6, lr=3e-4, ent=0.01, clip=0.2, scen=0.4),
    "explore harder": dict(frames=5e6, lr=1e-4, ent=0.03, clip=0.2, scen=0.6),
}


class Setup:
    def __init__(self, ckpts, current: int):
        self.ckpts = ckpts
        self.start = current if ckpts else -1          # index into ckpts, -1 = fresh network
        self.vals = {k[0]: k[3] for k in KNOBS}
        self.row = 0                                   # 0 = start-from, 1 = preset, then the knobs
        self.preset = 0
        self.name = "lab_" + time.strftime("%H%M%S")

    @property
    def rows(self) -> int:
        return 2 + len(KNOBS)

    def start_label(self) -> str:
        if self.start < 0:
            return "fresh network (random weights)"
        c = self.ckpts[self.start]
        return f"continue {c['run']}/{c['name']}"

    def change(self, d: int) -> None:
        if self.row == 0:
            n = len(self.ckpts) + 1
            self.start = (self.start + 1 + d) % n - 1
        elif self.row == 1:
            names = list(PRESETS)
            self.preset = (self.preset + d) % len(names)
            self.vals.update(PRESETS[names[self.preset]])
            if "fresh" in names[self.preset]:
                self.start = -1
        else:
            k = KNOBS[self.row - 2]
            vs = k[2]
            i = min(range(len(vs)), key=lambda j: abs(vs[j] - self.vals[k[0]]))
            self.vals[k[0]] = vs[max(0, min(len(vs) - 1, i + d))]

    def active(self, key: str) -> bool:
        return not (key in NEW_ONLY and self.start >= 0)

    def command(self) -> list[str]:
        cmd = [sys.executable, "scripts/train_ppo.py", "--name", self.name]
        for key, _l, _v, _d, flag, _f, _h in KNOBS:
            if not self.active(key):
                continue
            v = self.vals[key]
            cmd += [flag, str(int(v)) if flag in ("--frames", "--max-frames", "--eval-every", "--n-envs", "--n-steps", "--batch-size", "--epochs", "--net-width", "--net-layers", "--seed") else repr(float(v))]
        if self.start >= 0:
            cmd += ["--resume", self.ckpts[self.start]["path"]]
        return cmd
