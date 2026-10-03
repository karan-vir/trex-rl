"""The v9 browser script (scripts/archive/chrome_dino_agent_v9.js) re-implemented on the faithful engine.

Purpose: validate the engine at the level of a whole agent. v9 played ~40 real episodes (median 2,431,
62% of deaths on cacti); if the engine is faithful, this port should land near those numbers.
It mirrors the JS logic: per-obstacle measured speed, width-aware jump lead, duck pressed once,
and the (harmful) "let go after 3 frames stuck at ground level" rule.
"""

from __future__ import annotations

from collections import deque

from trex.engine import GROUND_Y, T_START_X, T_WIDTH, MS_PER_FRAME

NOSE_X = T_START_X + T_WIDTH
GROUND_BOTTOM = GROUND_Y + 47


class V9Agent:
    def __init__(self, lead=12.0, arc_center=16.5, width_aware=True):
        self.lead, self.arc_center, self.width_aware = lead, arc_center, width_aware
        self.seen: dict[int, float] = {}
        self.own: dict[int, deque] = {}
        self.all: deque = deque(maxlen=60)
        self.jump_down = False
        self.jump_frame = 0
        self.duck_down = False
        self.duck_pressed = False
        self.ground_frames = 0
        self.frame = 0

    # ---- speed measurement (updateMotion / effectiveSpeed in the JS) ----
    def _measure(self, e, dt):
        for ob in e.obstacles:
            k = id(ob)
            if k in self.seen and 1 < dt < 100:
                sample = (self.seen[k] - ob.x, dt)
                self.all.append(sample)
                d = self.own.setdefault(k, deque(maxlen=20))
                d.append(sample)
            self.seen[k] = ob.x
        if len(self.seen) > 50:
            alive = {id(o) for o in e.obstacles}
            self.seen = {k: v for k, v in self.seen.items() if k in alive}
            self.own = {k: v for k, v in self.own.items() if k in alive}

    @staticmethod
    def _rate(samples):
        dx = sum(s[0] for s in samples); dt = sum(s[1] for s in samples)
        return dx / dt * MS_PER_FRAME if dt > 0 else None

    def _eff_speed(self, e, ob):
        nominal = e.speed
        own = self.own.get(id(ob)) if ob else None
        if own and len(own) >= 8:
            v = self._rate(own)
        elif len(self.all) >= 10:
            v = self._rate(self.all)
        else:
            v = nominal * 0.9
        return min(nominal * 1.05, max(nominal * 0.5, v))

    # ---- decide() ----
    def decide(self, e):
        ahead = [o for o in e.obstacles if o.x + o.width > T_START_X]
        if not ahead:
            return "NOOP"
        o = ahead[0]
        speed = self._eff_speed(e, o)
        dist = o.x - NOSE_X
        fly_y = GROUND_BOTTOM - (o.y + o.height)
        arrives = dist / speed
        if fly_y >= 45:
            return "NOOP"
        if fly_y >= 20:
            return "DUCK" if arrives < self.lead else "NOOP"
        l = self.arc_center - (o.width + 44 - 8) / (2 * speed) if self.width_aware else self.lead
        if (not e.jumping) and 0 <= arrives < l:
            return "JUMP"
        return "NOOP"

    # ---- apply() ----
    def act(self, e, dt):
        """Called once per display frame BEFORE e.frame(); sends key events like the JS does."""
        self.frame += 1
        self._measure(e, dt)
        action = self.decide(e)
        # jump
        if self.jump_down and not e.jumping and self.frame - self.jump_frame >= 2:
            e.key_up("jump"); self.jump_down = False
        if action == "JUMP" and not self.jump_down and not e.jumping:
            e.key_down("jump"); self.jump_down = True; self.jump_frame = self.frame
        # duck (v9: once per jump, and let go after 3 frames stuck at ground level)
        at_ground_jumping = e.jumping and e.y >= GROUND_Y - 1
        self.ground_frames = self.ground_frames + 1 if at_ground_jumping else 0
        if not e.jumping:
            self.duck_pressed = False
        if action == "DUCK":
            if e.jumping:
                if not self.duck_pressed:
                    e.key_down("duck"); self.duck_pressed = True; self.duck_down = True
                elif self.ground_frames >= 3 and self.duck_down:
                    e.key_up("duck"); self.duck_down = False
            else:
                if not e.ducking:
                    e.key_down("duck")
                self.duck_down = True
        elif self.duck_down:
            e.key_up("duck"); self.duck_down = False
        return action
