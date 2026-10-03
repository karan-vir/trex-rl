"""The training lab: watch, poke at, and play against the agents.   python scripts/lab.py"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pygame

from trex.lab import draw
from trex.lab.session import ACTION_NAMES, Config, Session, find_checkpoints

W, H = 1280, 790
SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
HELP = [
    "1 you play   2 hand-written rule   3 trained network   4 random      [ ]  older / newer checkpoint      V  race the rule",
    "P pause   .  one step   + / -  speed      H display Hz   J jitter   L input delay   S dangerous starts      N new game   Q replay   C boxes",
    "T train from this checkpoint in the background      A auto-restart      F fullscreen      Esc quit        (you: up/space = jump, down = duck)",
]


class Lab:
    def __init__(self, seed: int = 0):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H), pygame.SCALED | pygame.RESIZABLE)
        pygame.display.set_caption("T-Rex RL lab")
        self.f = draw.fonts()
        self.cfg = Config(seed=seed)
        self.ckpts = find_checkpoints()
        self.ci = self._default_ckpt()
        self.models: dict = {}
        self.kind = "ppo" if self.ckpts else "rule"
        self.speed_i = SPEEDS.index(1)
        self.paused = False
        self.boxes = False
        self.auto = True
        self.race = False
        self.acc = 0.0
        self.done_at = None
        self.scores: dict = {}
        self.job = None
        self.job_name = None
        self.status = "ready"
        self.fullscreen = False
        self._build()

    # ------------------------------------------------------------------
    def _default_ckpt(self) -> int:
        for i in range(len(self.ckpts) - 1, -1, -1):
            if self.ckpts[i]["name"] == "best":
                return i
        return max(0, len(self.ckpts) - 1)

    def _model(self):
        if not self.ckpts:
            return None, ""
        c = self.ckpts[self.ci]
        if c["path"] not in self.models:
            from stable_baselines3 import PPO
            self.models[c["path"]] = PPO.load(c["path"], device="cpu")
        return self.models[c["path"]], f"{c['run']}/{c['name']}"

    def agent_label(self) -> str:
        if self.kind == "ppo":
            return "trained network: " + self._model()[1]
        return {"human": "you (keyboard)", "rule": "hand-written rule (v9)", "random": "random keys"}[self.kind]

    def _build(self) -> None:
        model, name = self._model() if self.kind == "ppo" else (None, "")
        if self.kind == "ppo" and model is None:
            self.kind = "rule"
        self.a = Session(self.kind, self.cfg, model, name)
        self.b = Session("rule", Config(**vars(self.cfg)), None, "") if self.race else None
        self.done_at = None
        self.acc = 0.0

    def new_game(self, seed=None) -> None:
        self.cfg.seed = self.cfg.seed + 1 if seed is None else seed
        self.a.reset(self.cfg.seed)
        if self.b:
            self.b.cfg.__dict__.update(vars(self.cfg))
            self.b.reset(self.cfg.seed)
        self.done_at = None

    # ------------------------------------------------------------------
    def key(self, k: int) -> bool:
        """Handle one key press. Returns False to quit."""
        m = {pygame.K_1: "human", pygame.K_2: "rule", pygame.K_3: "ppo", pygame.K_4: "random"}
        if k == pygame.K_ESCAPE:
            return False
        if k in m:
            self.kind = m[k]; self._build()
        elif k == pygame.K_LEFTBRACKET and self.ckpts:
            self.ci = (self.ci - 1) % len(self.ckpts); self.kind = "ppo"; self._build()
        elif k == pygame.K_RIGHTBRACKET and self.ckpts:
            self.ci = (self.ci + 1) % len(self.ckpts); self.kind = "ppo"; self._build()
        elif k == pygame.K_p:
            self.paused = not self.paused
        elif k == pygame.K_PERIOD:
            self.paused = True; self.a.step(self._human()); self.b and self.b.step()
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed_i = min(len(SPEEDS) - 1, self.speed_i + 1)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_i = max(0, self.speed_i - 1)
        elif k == pygame.K_h:
            hz = [60.0, 120.0, 144.0]; self.cfg.hz = hz[(hz.index(self.cfg.hz) + 1) % 3] if self.cfg.hz in hz else 120.0; self.new_game(self.cfg.seed)
        elif k == pygame.K_j:
            js = [0.0, 0.3, 0.6]; self.cfg.jitter = js[(js.index(self.cfg.jitter) + 1) % 3] if self.cfg.jitter in js else 0.3; self.new_game(self.cfg.seed)
        elif k == pygame.K_l:
            self.cfg.latency = 1 - self.cfg.latency; self.new_game(self.cfg.seed)
        elif k == pygame.K_s:
            self.cfg.scenario = not self.cfg.scenario; self.new_game(self.cfg.seed)
        elif k == pygame.K_n:
            self.new_game()
        elif k == pygame.K_q:
            self.new_game(self.cfg.seed)
        elif k == pygame.K_c:
            self.boxes = not self.boxes
        elif k == pygame.K_a:
            self.auto = not self.auto
        elif k == pygame.K_v:
            self.race = not self.race; self._build()
        elif k == pygame.K_t:
            self.toggle_training()
        elif k == pygame.K_f:
            self.fullscreen = not self.fullscreen; pygame.display.toggle_fullscreen()
        return True

    def _human(self) -> int:
        if self.kind != "human":
            return 0
        keys = pygame.key.get_pressed()
        return 1 if (keys[pygame.K_UP] or keys[pygame.K_SPACE]) else 2 if keys[pygame.K_DOWN] else 0

    # ------------------------------------------------------------------
    def toggle_training(self) -> None:
        if self.job and self.job.poll() is None:
            self.job.terminate(); self.status = f"training job {self.job_name} stopped"; return
        c = self.ckpts[self.ci] if self.ckpts else None
        self.job_name = "lab_" + time.strftime("%H%M%S")
        cmd = [sys.executable, "scripts/train_ppo.py", "--frames", "3000000", "--name", self.job_name, "--eval-every", "500000",
               "--eval-episodes", "12", "--max-frames", "20000", "--scenario-prob", "0.4", "--lr", "5e-5", "--ent-coef", "0.003", "--clip", "0.1"]
        if c:
            cmd += ["--resume", c["path"]]
        self.job = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.status = f"training {self.job_name}: 3M more decisions, evaluates every 0.5M ({'from ' + c['run'] + '/' + c['name'] if c else 'from scratch'})"

    def refresh_job(self) -> None:
        if self.job and self.job.poll() is not None and self.job_name and "done" not in self.status:
            self.ckpts = find_checkpoints()
            self.status = f"training {self.job_name} finished; new checkpoints loaded (use [ and ] to try them)"

    # ------------------------------------------------------------------
    def update(self, wall_dt: float) -> None:
        if not self.paused:
            self.acc += wall_dt * 60.0 * SPEEDS[self.speed_i]
            n = min(int(self.acc), 600)
            self.acc -= int(self.acc)
            for _ in range(n):
                self.a.step(self._human())
                if self.b:
                    self.b.step()
                if self.a.done and (self.b is None or self.b.done):
                    break
        both_done = self.a.done and (self.b is None or self.b.done)
        if both_done and self.done_at is None:
            self.done_at = time.time()
            self.scores.setdefault(self.agent_label(), []).append(self.a.score)
        if both_done and self.auto and self.done_at and time.time() - self.done_at > 1.4:
            self.new_game()
        self.refresh_job()

    # ------------------------------------------------------------------
    def render(self) -> None:
        s, f = self.screen, self.f
        s.fill(draw.BG)
        draw.text(s, f["xl"], "T-Rex RL lab", (20, 12))
        draw.text(s, f["m"], self.agent_label(), (210, 17), draw.BLUE)
        c = self.cfg
        chips = (f"speed {SPEEDS[self.speed_i]}x{'  PAUSED' if self.paused else ''}   display {c.hz:.0f} Hz   jitter {c.jitter} ms   "
                 f"input delay {c.latency}   dangerous starts {'ON' if c.scenario else 'off'}   seed {c.seed}")
        draw.text(s, f["s"], chips, (W - 20, 20), draw.MUTE, "right")
        sc = self.scores.get(self.agent_label(), [])
        stat = f"this agent so far: {len(sc)} games" + (f", median {int(np.median(sc))}, best {max(sc)}" if sc else "")
        draw.text(s, f["s"], stat, (20, 40), draw.MUTE)

        if self.race:
            draw.draw_game(s, self.a, 20, 60, 1, self.boxes, self.agent_label(), f)
            draw.draw_game(s, self.b, 660, 60, 1, self.boxes, "hand-written rule (v9), same course", f)
            top = 235
        else:
            draw.draw_game(s, self.a, 40, 60, 2, self.boxes, "", f)
            top = 375
        h = H - 85 - top
        draw.draw_inputs(s, pygame.Rect(20, top, 520, h), self.a, f)
        draw.draw_actions(s, pygame.Rect(550, top, 330, 170), self.a, f)
        draw.draw_events(s, pygame.Rect(550, top + 180, 330, h - 180), self.a, f)
        draw.draw_brain(s, pygame.Rect(890, top, 370, min(210, h - 140)), self.a, f)
        run = self.job_name if (self.job and self.job_name) else (self.ckpts[self.ci]["run"] if self.ckpts else "")
        marker = self.ckpts[self.ci]["M"] if (self.ckpts and not self.job) else None
        title = f"LEARNING CURVE ({run})" + (" - live" if self.job and self.job.poll() is None else "")
        draw.draw_curve(s, pygame.Rect(890, top + min(210, h - 140) + 10, 370, h - min(210, h - 140) - 10), draw.load_curve(run), marker, title, f)

        y = H - 78
        for line in HELP:
            draw.text(s, f["s"], line, (20, y), draw.MUTE); y += 15
        if self.status:
            draw.text(s, f["s"], self.status[:150], (20, H - 22), draw.ORANGE)

    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    running = self.key(ev.key) and running
            self.update(clock.tick(60) / 1000.0)
            self.render()
            pygame.display.flip()
        if self.job and self.job.poll() is None:
            self.job.terminate()
        pygame.quit()


def main():
    Lab().run()
