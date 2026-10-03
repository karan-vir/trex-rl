"""A faithful port of the REAL game engine (chromedino.com, js/game.js v29).

Why this exists: the earlier simulator (trex/game.py) approximated the game and kept being wrong in
ways that only showed up on the real thing (stuck landings, a speed quirk, wrong collision shapes).
This module copies the real code's logic line by line, so an agent trained here sees the same
mechanics as the real game, including its quirks:

  * the clock is `new Date().getTime()` (integer milliseconds), so at 120 frames/s the frame time
    alternates between 8 and 9 ms; obstacles move by floor(speed * 0.06 * dt) whole pixels;
  * speed grows by 0.001 on EVERY update call (not per unit of time);
  * the dino's jump uses the frame time of its current animation STATE (jumping 16.7 ms, running
    83 ms, ducking 125 ms); a fast fall that lands exactly on the ground turns into a duck while the
    game still thinks the dino is jumping, and the dino then falls ~7.5x slower (a ~1 s stuck state);
  * only obstacles[0] is collision-tested, with the real multi-box shapes;
  * keys are EVENTS (keydown/keyup), and a repeated keydown(duck) restarts the fast fall.

Coordinates are the game's own: y grows DOWNWARD, the dino's yPos is its top (93 on the ground).
"""

from __future__ import annotations

import math
import random

# --- constants copied from game.js -------------------------------------------------------
FPS = 60
MS_PER_FRAME = 1000.0 / FPS
WIDTH, HEIGHT = 600, 150
SPEED0, MAX_SPEED, ACCELERATION = 6.0, 13.0, 0.001
GRAVITY, SPEED_DROP_COEFFICIENT = 0.6, 3.0
CLEAR_TIME = 3000
GAP_COEFFICIENT, MAX_GAP_COEFFICIENT = 0.6, 1.5
MAX_OBSTACLE_LENGTH, MAX_OBSTACLE_DUPLICATION = 3, 2

T_WIDTH, T_HEIGHT, T_START_X = 44, 47, 50
T_INITIAL_JUMP_VELOCITY, T_DROP_VELOCITY = -10.0, -5.0
T_MAX_JUMP_HEIGHT, T_MIN_JUMP_HEIGHT = 30, 30
GROUND_Y = HEIGHT - T_HEIGHT - 10                     # 93
MIN_JUMP_Y = GROUND_Y - T_MIN_JUMP_HEIGHT             # 63

RUNNING, JUMPING, DUCKING = "RUNNING", "JUMPING", "DUCKING"
ANIM_MS = {RUNNING: 1000 / 12, JUMPING: 1000 / 60, DUCKING: 1000 / 8}

TREX_BOXES_RUNNING = ((22, 0, 17, 16), (1, 18, 30, 9), (10, 35, 14, 8),
                      (1, 24, 29, 5), (5, 30, 21, 4), (9, 34, 15, 4))
TREX_BOXES_DUCKING = ((1, 18, 55, 25),)

# type: (name, width, height, yPos(s), multipleSpeed, minGap, minSpeed, boxes, speedOffset)
CACTUS_SMALL, CACTUS_LARGE, PTERODACTYL = 0, 1, 2
OBSTACLE_TYPES = (
    ("CACTUS_SMALL", 17, 35, (HEIGHT - 55,), 4, 120, 0.0, ((0, 7, 5, 27), (4, 0, 6, 34), (10, 4, 7, 14)), 0.0),
    ("CACTUS_LARGE", 25, 50, (HEIGHT - 60,), 7, 120, 0.0, ((0, 12, 7, 38), (8, 0, 7, 49), (13, 10, 10, 38)), 0.0),
    ("PTERODACTYL", 46, 40, (HEIGHT - 50, HEIGHT - 75, HEIGHT - 100), 999, 150, 8.5,
     ((15, 15, 16, 5), (18, 21, 24, 6), (2, 14, 4, 3), (6, 10, 4, 7), (10, 8, 6, 9)), 0.8),
)


def jround(x: float) -> int:
    """JavaScript Math.round: ties go toward +infinity."""
    return math.floor(x + 0.5)


class Obstacle:
    __slots__ = ("kind", "size", "x", "y", "width", "height", "gap", "offset", "boxes", "following", "remove")

    def __init__(self, kind: int, rng: random.Random, speed: float):
        name, w, h, ys, multiple_speed, min_gap, _, boxes, offset = OBSTACLE_TYPES[kind]
        self.kind = kind
        self.size = rng.randint(1, MAX_OBSTACLE_LENGTH)
        if self.size > 1 and multiple_speed > speed:
            self.size = 1
        self.width = w * self.size
        self.height = h
        self.x = WIDTH - self.width
        self.y = ys[rng.randint(0, len(ys) - 1)] if len(ys) > 1 else ys[0]
        b = [list(bx) for bx in boxes]
        if self.size > 1:
            b[1][2] = self.width - b[0][2] - b[2][2]
            b[2][0] = self.width - b[2][2]
        self.boxes = b
        self.offset = (offset if rng.random() > 0.5 else -offset) if offset else 0.0
        lo = jround(self.width * speed + min_gap * GAP_COEFFICIENT)
        hi = jround(lo * MAX_GAP_COEFFICIENT)
        self.gap = rng.randint(lo, hi)
        self.following = False
        self.remove = False

    def update(self, dt: int, speed: float) -> None:
        if not self.remove:
            if self.offset:
                speed += self.offset
            self.x -= math.floor((speed * FPS / 1000) * dt)
            if not (self.x + self.width > 0):
                self.remove = True


class RealEngine:
    """One game. Call key_down / key_up for input, then frame() once per display frame."""

    def __init__(self, seed=None, frame_ms: float = 1000 / 120, jitter_ms: float = 0.0,
                 start_speed: float = SPEED0, skip_clear_time: bool = False):
        self.rng = random.Random(seed)
        self.frame_ms = frame_ms
        self.jitter_ms = jitter_ms
        self.true_ms = self.rng.uniform(0.0, 1000.0)      # unknown phase of the integer-ms clock
        self.speed = start_speed
        self.running_time = float(CLEAR_TIME + 1) if skip_clear_time else 0.0
        self.distance = 0.0
        self.crashed = False
        self.frames = 0
        # dino
        self.y = GROUND_Y
        self.x = T_START_X
        self.vel = 0.0
        self.jumping = False
        self.ducking = False
        self.speed_drop = False
        self.reached_min = False
        self.status = RUNNING
        # horizon
        self.obstacles: list[Obstacle] = []
        self.history: list[int] = []
        self.time = self._now()

    # ---- clock ------------------------------------------------------------------------
    def _now(self) -> int:
        return int(math.floor(self.true_ms))               # new Date().getTime()

    # ---- the dino (Trex) -----------------------------------------------------------------
    def _trex_update(self, status=None) -> None:
        if status:
            self.status = status
        if self.speed_drop and self.y == GROUND_Y:         # "speed drop becomes duck if the key is held"
            self.speed_drop = False
            self._set_duck(True)

    def _set_duck(self, is_ducking: bool) -> None:
        if is_ducking and self.status != DUCKING:
            self._trex_update(DUCKING)
            self.ducking = True
        elif self.status == DUCKING:                       # NB: also fires for is_ducking=True (toggles)
            self._trex_update(RUNNING)
            self.ducking = False

    def _start_jump(self) -> None:
        if not self.jumping:
            self._trex_update(JUMPING)
            self.vel = T_INITIAL_JUMP_VELOCITY - (self.speed / 10)
            self.jumping = True
            self.reached_min = False
            self.speed_drop = False

    def _end_jump(self) -> None:
        if self.reached_min and self.vel < T_DROP_VELOCITY:
            self.vel = T_DROP_VELOCITY

    def _reset_dino(self) -> None:
        self.y = GROUND_Y
        self.vel = 0.0
        self.jumping = False
        self.ducking = False
        self._trex_update(RUNNING)
        self.speed_drop = False

    def _update_jump(self, dt: int) -> None:
        fe = dt / ANIM_MS[self.status]                     # frames elapsed, in the CURRENT state's frame time
        if self.speed_drop:
            self.y += jround(self.vel * SPEED_DROP_COEFFICIENT * fe)
        else:
            self.y += jround(self.vel * fe)
        self.vel += GRAVITY * fe
        if self.y < MIN_JUMP_Y or self.speed_drop:
            self.reached_min = True
        if self.y < T_MAX_JUMP_HEIGHT or self.speed_drop:
            self._end_jump()
        if self.y > GROUND_Y:
            self._reset_dino()
        self._trex_update()

    # ---- keys (the game's onKeyDown / onKeyUp) ----------------------------------------------
    def key_down(self, key: str) -> None:
        if self.crashed:
            return
        if key == "jump":
            if not self.jumping and not self.ducking:
                self._start_jump()
        elif key == "duck":
            if self.jumping:
                self.speed_drop = True                     # setSpeedDrop
                self.vel = 1.0
            elif not self.ducking:
                self._set_duck(True)

    def key_up(self, key: str) -> None:
        if self.crashed:
            return
        if key == "jump":
            self._end_jump()
        elif key == "duck":
            self.speed_drop = False
            self._set_duck(False)

    # ---- obstacles (Horizon) -----------------------------------------------------------------
    def _add_obstacle(self) -> None:
        while True:
            kind = self.rng.randint(0, len(OBSTACLE_TYPES) - 1)
            dup = 0
            for h in self.history:
                dup = dup + 1 if h == kind else 0
            if dup >= MAX_OBSTACLE_DUPLICATION or self.speed < OBSTACLE_TYPES[kind][6]:
                continue
            break
        self.obstacles.append(Obstacle(kind, self.rng, self.speed))
        self.history.insert(0, kind)
        del self.history[MAX_OBSTACLE_DUPLICATION:]

    def _update_obstacles(self, dt: int) -> None:
        keep = self.obstacles[:]
        for ob in self.obstacles:
            ob.update(dt, self.speed)
            if ob.remove:
                keep.pop(0)
        self.obstacles = keep
        if self.obstacles:
            last = self.obstacles[-1]
            if (not last.following and last.x + last.width > 0
                    and last.x + last.width + last.gap < WIDTH):
                self._add_obstacle()
                last.following = True
        else:
            self._add_obstacle()

    def inject(self, kind: int, x: float, y_index: int = 0, size: int = 1) -> Obstacle:
        """Place an obstacle at a chosen position (used to create rare situations for training)."""
        ob = Obstacle(kind, self.rng, self.speed)
        ob.size = size if (size == 1 or OBSTACLE_TYPES[kind][4] <= self.speed) else 1
        name, w, h, ys, multiple_speed, min_gap, _, boxes, offset = OBSTACLE_TYPES[kind]
        ob.width = w * ob.size
        ob.y = ys[min(y_index, len(ys) - 1)]
        b = [list(bx) for bx in boxes]
        if ob.size > 1:
            b[1][2] = ob.width - b[0][2] - b[2][2]
            b[2][0] = ob.width - b[2][2]
        ob.boxes = b
        ob.x = x
        self.obstacles.append(ob)
        self.obstacles.sort(key=lambda o: o.x)
        self.history.insert(0, kind)
        del self.history[MAX_OBSTACLE_DUPLICATION:]
        return ob

    # ---- collisions -------------------------------------------------------------------------
    def _collides(self, ob: Obstacle) -> bool:
        tx, ty, tw, th = self.x + 1, self.y + 1, T_WIDTH - 2, T_HEIGHT - 2
        ox, oy, ow, oh = ob.x + 1, ob.y + 1, ob.width - 2, ob.height - 2
        if not (tx < ox + ow and tx + tw > ox and ty < oy + oh and th + ty > oy):
            return False
        for (bx, by, bw, bh) in (TREX_BOXES_DUCKING if self.ducking else TREX_BOXES_RUNNING):
            ax, ay = bx + tx, by + ty
            for (cx, cy, cw, ch) in ob.boxes:
                px, py = cx + ox, cy + oy
                if ax < px + cw and ax + bw > px and ay < py + ch and bh + ay > py:
                    return True
        return False

    # ---- one display frame (Runner.update) -----------------------------------------------------
    def frame(self) -> bool:
        """Advance one display frame. Returns False once the dino has crashed."""
        if self.crashed:
            return False
        self.true_ms += self.frame_ms + (self.rng.gauss(0.0, self.jitter_ms) if self.jitter_ms else 0.0)
        now = self._now()
        dt = now - self.time
        self.time = now
        self.frames += 1
        if self.jumping:
            self._update_jump(dt)
        self.running_time += dt
        has_obstacles = self.running_time > CLEAR_TIME
        if has_obstacles:
            self._update_obstacles(dt)
        if has_obstacles and self.obstacles and self._collides(self.obstacles[0]):
            self.crashed = True
            return False
        self.distance += self.speed * dt / MS_PER_FRAME
        if self.speed < MAX_SPEED:
            self.speed += ACCELERATION
        self.last_dt = dt
        return True

    # ---- read-outs ----------------------------------------------------------------------------
    @property
    def height(self) -> float:
        """Dino height above the ground in px (0 on the ground)."""
        return GROUND_Y - self.y

    @property
    def score(self) -> int:
        return jround(math.ceil(self.distance) * 0.025)
