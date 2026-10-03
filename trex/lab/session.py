"""One playing session for the lab: an agent (human / random / rule / trained network) on the faithful engine.

A "decision" is the lab's time step: about 1/60 s of game time (1 display frame at 60 Hz, 2 at 120 Hz).
The session also records what the agent is "thinking" (the 22 inputs it sees, the probability it gives
each action, and the activity of its hidden layers) so the UI can show it.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trex.agents.v9_on_engine import V9Agent
from trex.engine import GROUND_Y, JUMPING, PTERODACTYL, RealEngine
from trex.real_env import OBS_DIM, RealTRexEnv

ACTION_NAMES = ["no key", "JUMP held", "DUCK held"]
OBS_NAMES = (
    ["height", "vertical speed", "jumping", "ducking", "fast falling", "duck-anim while jumping",
     "game speed", "observed speed", "frame time"]
    + [f"obstacle {k + 1} {n}" for k in range(2) for n in ("distance", "width", "height", "flying height", "is pterodactyl")]
    + ["holding: nothing", "holding: jump", "holding: duck"]
)


@dataclass
class Config:
    hz: float = 120.0
    jitter: float = 0.3
    latency: int = 0
    scenario: bool = False
    seed: int = 0


def find_checkpoints(root: Path = Path("runs")):
    """Every saved PPO model, grouped by run and ordered by training progress."""
    out = []
    for run in sorted(p for p in root.glob("*") if p.is_dir()):
        files = list(run.glob("*.zip"))
        def key(f):
            m = re.match(r"ckpt_(\d+)M", f.stem)
            return (0, int(m.group(1))) if m else (1, {"latest": 1, "best": 2, "final": 3}.get(f.stem, 4))
        for f in sorted(files, key=key):
            m = re.match(r"ckpt_(\d+)M", f.stem)
            out.append({"run": run.name, "name": f.stem, "path": str(f), "M": int(m.group(1)) if m else None})
    return out


class Session:
    def __init__(self, kind: str, cfg: Config, model=None, ckpt_name: str = ""):
        self.kind, self.cfg, self.model, self.ckpt_name = kind, cfg, model, ckpt_name
        self.events: deque = deque(maxlen=10)
        self.rng = np.random.default_rng(cfg.seed)
        self.reset(cfg.seed)

    # ------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        self.cfg.seed = seed
        c = self.cfg
        frame_ms = 1000.0 / c.hz
        self.done, self.crash_text, self.decisions = False, "", 0
        self.last_action, self.held = 0, 0
        self.probs = None
        self.h1 = self.h2 = None
        self.obs = np.zeros(OBS_DIM, dtype=np.float32)
        self.passed: set[int] = set()
        self.rule = None
        if self.kind == "rule":
            self.engine = RealEngine(seed=seed, frame_ms=frame_ms, jitter_ms=c.jitter)
            self.rule = V9Agent()
            self.engine.frame()
            self.skip = max(1, round(1000 / 60 / frame_ms))
            self.rule_action = "NOOP"
            self.env = None
        else:
            self.env = RealTRexEnv(randomize=False, frame_ms=frame_ms, jitter_ms=c.jitter, latency=c.latency,
                                   scenario_prob=1.0 if c.scenario else 0.0, max_frames=10 ** 9)
            self.obs, _ = self.env.reset(seed=seed)
            self.engine = self.env.engine
        self._snapshot()
        self.events.append(f"new game, seed {seed}, {c.hz:.0f} Hz")

    def _snapshot(self) -> None:
        e = self.engine
        self.prev = (e.jumping, e.speed_drop, e.status, bool(e.ducking), self.held)

    # ------------------------------------------------------------------
    def step(self, human_action: int = 0) -> None:
        """Advance one decision (about 1/60 s of game time)."""
        if self.done:
            return
        e = self.engine
        if self.kind == "rule":
            alive = True
            for _ in range(self.skip):
                a = self.rule.act(e, e.last_dt)
                alive = e.frame()
                if not alive:
                    break
            self.rule_action = a
            self.held = 1 if self.rule.jump_down else 2 if self.rule.duck_down else 0
            self.last_action = self.held
            self.decisions += 1
            if not alive:
                self._finish()
        else:
            if self.kind == "human":
                act = human_action
            elif self.kind == "random":
                act = self.last_action if self.rng.random() < 0.9 else int(self.rng.integers(3))
            else:                                                     # trained network
                act, self.probs, self.h1, self.h2 = self._think(self.obs)
            self.last_action = int(act)
            self.obs, r, te, tu, info = self.env.step(int(act))
            self.held = self.env.held
            self.decisions += 1
            if te:
                self._finish()
        self._events()
        self._snapshot()

    def _think(self, obs):
        import torch
        pol = self.model.policy
        with torch.no_grad():
            x = torch.as_tensor(obs[None])
            feats = pol.extract_features(x, pol.pi_features_extractor)
            h, acts = feats, []
            for layer in pol.mlp_extractor.policy_net:        # Linear, Tanh, Linear, Tanh, ...
                h = layer(h)
                if not isinstance(layer, torch.nn.Linear):
                    acts.append(h[0].numpy())
            probs = torch.softmax(pol.action_net(h), dim=-1)[0].numpy()
        return int(np.argmax(probs)), probs, acts[0], acts[-1] if len(acts) > 1 else None

    def _events(self) -> None:
        e = self.engine
        pj, psd, pstatus, pduck, pheld = self.prev
        if e.jumping and not pj:
            self.events.append(f"jump  (height gain starts, speed {e.speed:.1f})")
        if pj and not e.jumping:
            self.events.append("landed" + ("  (into a duck)" if e.ducking else ""))
        if e.speed_drop and not psd:
            self.events.append("fast fall started")
        if e.jumping and e.status != JUMPING and e.y == GROUND_Y and not (pj and pstatus != JUMPING):
            self.events.append("!! STUCK LANDING: on the ground but still 'jumping' (slow physics)")
        if self.held != pheld:
            self.events.append(f"keys: {ACTION_NAMES[pheld]}  ->  {ACTION_NAMES[self.held]}")
        for ob in e.obstacles:
            if ob.x + ob.width < 50 and id(ob) not in self.passed:
                self.passed.add(id(ob))
                self.events.append(f"passed a {'pterodactyl' if ob.kind == PTERODACTYL else 'cactus'} x{ob.size}")
        if len(self.passed) > 60:
            self.passed &= {id(o) for o in e.obstacles}

    def _finish(self) -> None:
        self.done = True
        e = self.engine
        ob = e.obstacles[0] if e.obstacles else None
        kind = ("pterodactyl" if ob.kind == PTERODACTYL else "cactus") + f" x{ob.size}" if ob else "?"
        self.crash_text = f"CRASH into a {kind}  -  score {e.score}"
        self.events.append(self.crash_text)

    @property
    def score(self) -> int:
        return self.engine.score
