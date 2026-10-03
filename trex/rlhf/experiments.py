"""Experiment cards: ready-made RLHF experiments. Each runs the WHOLE pipeline per variant:

    judge N clip pairs (hidden judge, with optional label noise) -> train a reward model -> RL against it -> grade on the real score

so you can change ONE thing and see what it does. Results are saved to runs/rlhf/exp/<card>.json.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EXP_DIR = Path("runs/rlhf/exp")
BASE = dict(labels=600, noise=0.0, beta=0.05, frames=1_500_000, scale=0.1, shift=0.0, ref="runs/ppo_d/ckpt_007M")

CARDS = [
    dict(id="how_many", title="How many human judgments do you really need?", icon="1",
         question="The reward model is trained from your comparisons. If you only had a few, would RL still improve the dino?",
         predict=["Barely any: a few judgments aren't enough", "A little: it improves but stays flaky", "Already enough: more judgments don't help much"],
         tells="Look at 'RM agreement' (does the reward model agree with the hidden judge on new clips?) and at 'worst 10% of games'.",
         llm="Preference datasets are expensive. How labels scale vs. policy quality is a central budget question in post-training.",
         variants=[dict(label="30 judgments", labels=30), dict(label="300 judgments", labels=300), dict(label="1500 judgments", labels=1500)]),
    dict(id="noisy", title="What if the judges are sometimes wrong?", icon="2",
         question="Real people disagree and make mistakes. We flip a share of the labels at random. How much damage does that do?",
         predict=["Little: the model averages the noise out", "Some: it gets noticeably worse", "A lot: 20% wrong labels already breaks it"],
         tells="Compare RM agreement and the final grade as noise grows.",
         llm="Annotator disagreement is typically 20-30% on open-ended tasks. It caps how good a reward model can be.",
         variants=[dict(label="0% wrong", noise=0.0), dict(label="20% wrong", noise=0.2), dict(label="40% wrong", noise=0.4)]),
    dict(id="leash", title="What does the KL leash do?", icon="3",
         question="beta is the penalty for drifting away from the reference model. Set it to zero, medium, or very high. What changes?",
         predict=["No leash is best: more freedom, higher score", "A medium leash is best", "A very strong leash is best"],
         tells="Watch 'drift from reference (KL)': how far each policy moved, versus the grade it earned.",
         llm="The KL term in RLHF keeps the model fluent and stops it from exploiting the reward model. Tuning beta is a core knob.",
         variants=[dict(label="beta 0 (no leash)", beta=0.0), dict(label="beta 0.05", beta=0.05), dict(label="beta 1.0 (tight leash)", beta=1.0)]),
    dict(id="overopt", title="What if you keep optimising the same reward model?", icon="4",
         question="We train 4x longer against a reward model built from only 150 judgments. Does the agent keep getting better at the real game, or does it learn to please the reward model?",
         predict=["Keeps improving on the real score", "Improves, then plateaus", "Improves, then gets WORSE on the real score"],
         tells="On the chart, compare the reward the agent is paid (proxy) with the real score over time. Divergence means over-optimization.",
         llm="Reward over-optimization (Goodhart's law): past some point, more RL against a proxy reward hurts true quality. Gao et al. measure this as a scaling law.",
         variants=[dict(label="150 judgments, 6M steps", labels=150, frames=6_000_000)]),
]
BY_ID = {c["id"]: c for c in CARDS}


def read_rows(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def run_card(card_id: str, log=print) -> None:
    from stable_baselines3 import PPO
    from trex.rlhf.core import make_pair
    from trex.rlhf.reward_model import accuracy_vs_judge, save_rm, train_reward_model

    card = BY_ID[card_id]
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXP_DIR / f"{card_id}.json"
    res = dict(card=card_id, started=time.time(), done=False, variants=[], stage="starting")
    save = lambda: out_path.write_text(json.dumps(res))
    save()
    for vi, v in enumerate(card["variants"]):
        cfg = {**BASE, **v}
        name = f"exp_{card_id}_{vi}"
        entry = dict(label=v["label"], cfg=cfg, run=name, stage="collecting judgments")
        res["variants"].append(entry); res["stage"] = f"{v['label']}: collecting judgments"; save()
        ref = PPO.load(cfg["ref"], device="cpu")
        rng = np.random.default_rng(100 + vi)
        recs = []
        while len(recs) < cfg["labels"]:
            p = make_pair(ref, rng)
            lab = p["truth"]
            if lab != "same" and rng.random() < cfg["noise"]:
                lab = "B" if lab == "A" else "A"
            recs.append({"spec": p["spec"], "A": p["A"], "B": p["B"], "label": lab})
        entry["stage"] = "training reward model"; res["stage"] = f"{v['label']}: training reward model"; save()
        rm, hist, _ = train_reward_model(recs, epochs=60)
        rm_dir = EXP_DIR / name
        rm_dir.mkdir(parents=True, exist_ok=True)
        save_rm(rm, rm_dir / "rm.pt", {"labels": cfg["labels"]})
        entry["rm_agreement"] = accuracy_vs_judge(rm, ref, n=150)
        entry["stage"] = "RL against the reward model"; res["stage"] = f"{v['label']}: RL against the reward model"; save()
        cmd = [sys.executable, "-m", "scripts.train_rlhf", "--ref", cfg["ref"], "--rm", str(rm_dir / "rm.pt"), "--name", name,
               "--frames", str(int(cfg["frames"])), "--beta", str(cfg["beta"]), "--scale", str(cfg["scale"]), "--shift", str(cfg["shift"]),
               "--eval-every", str(int(max(250_000, cfg["frames"] // 12)))]
        with open(Path("runs") / f"{name}.log", "w") as lg:
            proc = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                time.sleep(5)
                rows = read_rows(Path("runs") / name / "eval.csv")
                entry["curve"] = rows; entry["proxy"] = read_rows(Path("runs") / name / "rlhf.csv")[-400:]
                save()
        entry["curve"] = read_rows(Path("runs") / name / "eval.csv")
        entry["proxy"] = read_rows(Path("runs") / name / "rlhf.csv")
        entry["stage"] = "done"
        save()
    res.update(done=True, stage="done", finished=time.time())
    save()
