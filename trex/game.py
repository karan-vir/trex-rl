"""The T-Rex Runner rules. No graphics and no Gymnasium in here.

The game advances in fixed "ticks" (think: one frame at 60 per second). Calling
`step(action)` always moves the world forward exactly one tick, so the same seed
and the same actions always give the same game. That property is what makes
training an AI practical.

Coordinates: x grows to the right, y is the height ABOVE the ground
(y = 0 means standing on the ground). The renderer flips this for the screen.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# --- Actions ---------------------------------------------------------------
NOOP, JUMP, DUCK = 0, 1, 2

# --- World -----------------------------------------------------------------
WORLD_WIDTH = 600
START_SPEED = 6.0       # pixels per tick the world scrolls left
MAX_SPEED = 13.0
SPEED_RAMP = 0.001      # speed gained per tick

# --- Dino ------------------------------------------------------------------
DINO_X = 50
DINO_W, DINO_H = 44, 47
DUCK_W, DUCK_H = 59, 25
GRAVITY = 0.6           # pixels/tick^2 pulling the dino down
JUMP_VELOCITY = 10.0    # initial upward speed of a jump
FAST_FALL = 3.0         # gravity multiplier while ducking in mid-air
HITBOX_SHRINK = 4       # hitboxes are smaller than sprites, like the real game

# --- Obstacles: kind -> (width, height) --------------------------------------
CACTUS_SMALL, CACTUS_LARGE, PTERO = "cactus_small", "cactus_large", "ptero"
OBSTACLE_SIZES = {
    CACTUS_SMALL: (17, 35),
    CACTUS_LARGE: (25, 50),
    PTERO: (46, 30),
}
# Pterodactyls fly at one of three heights above the ground:
#   8  -> must jump (ducking is not low enough)
#   30 -> must duck (standing hits it, ducking fits underneath)
#   60 -> flies over a standing dino, can be ignored
PTERO_HEIGHTS = (8, 30, 60)
PTERO_MIN_SCORE = 300   # no pterodactyls in the early game

# Gap between obstacles, as a multiple of the current speed (in ticks of travel).
# A full jump lasts ~33 ticks, so a minimum of 40 ticks keeps every obstacle clearable.
MIN_GAP_TICKS = 40
MAX_GAP_FACTOR = 2.2

# --- "Chrome-like" mode ------------------------------------------------------------
# TRexGame(chrome_like=True) swaps in the real game's obstacle generator and jump, as read
# from chromedino.com (the classic Chromium dino code). Differences from the simple mode:
#   * cacti come in groups of 1-3 (large groups only from speed 7, small from speed 4)
#   * the gap after an obstacle grows with ITS width and the speed:
#         gap = uniform(width * speed + min_gap * 0.6,  1.5 * that)       (px)
#     so after a narrow obstacle at high speed the next one can arrive only ~22 ticks later
#   * pterodactyls only once speed >= 8.5, flying at heights 0 / 25 / 50, 40 px tall
#   * no more than 2 identical obstacle types in a row
#   * first obstacle appears after 3 seconds (180 ticks)
#   * the jump gets stronger with speed (v0 = 10 + speed/10) and is capped: once above
#     63 px the rise slows to 5 px/tick, so the apex is ~85 px however fast you go
CHROME_SIZES = {CACTUS_SMALL: (17, 35), CACTUS_LARGE: (25, 50), PTERO: (46, 40)}
CHROME_MIN_GAP = {CACTUS_SMALL: 120, CACTUS_LARGE: 120, PTERO: 150}      # px, before x 0.6
CHROME_MULTIPLE_SPEED = {CACTUS_SMALL: 4.0, CACTUS_LARGE: 7.0, PTERO: 1e9}
CHROME_MIN_SPEED = {CACTUS_SMALL: 0.0, CACTUS_LARGE: 0.0, PTERO: 8.5}
CHROME_PTERO_HEIGHTS = (0, 25, 50)
GAP_COEFFICIENT, MAX_GAP_COEFFICIENT = 0.6, 1.5
MAX_CLUSTER, MAX_DUPLICATES = 3, 2
CLEAR_TICKS = 180
JUMP_SPEED_BONUS = 0.1
JUMP_CAP_HEIGHT, JUMP_CAP_VELOCITY = 63.0, 5.0


@dataclass
class Obstacle:
    kind: str
    x: float          # left edge
    y: float          # bottom edge, height above ground
    w: int            # total width (a group of 3 cacti is 3x the single width)
    h: int
    size: int = 1     # how many cacti are in the group


def _hitbox(x: float, y: float, w: float, h: float):
    """Shrink a box a little so near-misses don't count as hits."""
    s = HITBOX_SHRINK
    return (x + s, y + s, x + w - s, y + h - s)  # left, bottom, right, top


def _overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def dino_physics(
    y: float, vy: float, action: int, speed: float | None = None
) -> tuple[float, float, bool]:
    """One tick of dino physics. Returns (new_y, new_vy, is_ducking).

    A pure function (no game object needed), so tests and planning agents can
    ask "what happens if the dino does X from here?" without copying the game.
    Pass `speed` to get the Chrome-like jump (stronger at speed, capped apex).
    """
    on_ground = y <= 0
    if action == JUMP and on_ground:
        vy = JUMP_VELOCITY if speed is None else JUMP_VELOCITY + speed * JUMP_SPEED_BONUS
    # Ducking = holding DUCK on the ground. In mid-air the same key means fall faster.
    ducking = action == DUCK and on_ground
    fast_fall = action == DUCK and not on_ground

    if on_ground and vy <= 0:
        return 0.0, 0.0, ducking
    vy -= GRAVITY * (FAST_FALL if fast_fall else 1.0)
    y += vy
    if y <= 0:   # landed
        return 0.0, 0.0, ducking
    if speed is not None and y > JUMP_CAP_HEIGHT and vy > JUMP_CAP_VELOCITY:
        vy = JUMP_CAP_VELOCITY
    return y, vy, ducking


def dino_hitbox(y: float, ducking: bool):
    w, h = (DUCK_W, DUCK_H) if ducking else (DINO_W, DINO_H)
    return _hitbox(DINO_X, y, w, h)


class TRexGame:
    def __init__(self, seed: int | None = None, chrome_like: bool = False,
                 motion_scale: float = 1.0):
        self.chrome_like = chrome_like
        # Real-game quirk: obstacles can move slower than the reported speed. On a 120 Hz
        # screen chromedino.com moves them in whole pixels per frame (floor), measured at
        # ~0.86-0.87x the reported speed. The reported (observed) speed stays unchanged.
        self.motion_scale = motion_scale
        self.reset(seed)

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.speed = START_SPEED
        self.score = 0.0
        self.ticks = 0
        self.distance = 0.0       # total pixels scrolled (used for background animation)
        self.passed = 0           # obstacles successfully dodged
        self.done = False
        self.dino_y = 0.0
        self.dino_vy = 0.0
        self.ducking = False
        self.obstacles: list[Obstacle] = []
        self._recent: list[str] = []             # last obstacle kinds (chrome-like duplicate rule)
        # first obstacle shows up a bit late (3 seconds late in chrome-like mode)
        self._next_spawn_x = WORLD_WIDTH + (CLEAR_TICKS * START_SPEED if self.chrome_like else 200)

    # ------------------------------------------------------------------
    @property
    def on_ground(self) -> bool:
        return self.dino_y <= 0

    @property
    def dino_size(self):
        return (DUCK_W, DUCK_H) if self.ducking else (DINO_W, DINO_H)

    def dino_box(self):
        return dino_hitbox(self.dino_y, self.ducking)

    def next_obstacles(self, n: int = 2) -> list[Obstacle]:
        """The nearest obstacles still ahead of (or touching) the dino."""
        ahead = [o for o in self.obstacles if o.x + o.w > DINO_X]
        return ahead[:n]

    # ------------------------------------------------------------------
    def step(self, action: int) -> bool:
        """Advance one tick. Returns True if the dino is still alive."""
        if self.done:
            return False

        self.dino_y, self.dino_vy, self.ducking = dino_physics(
            self.dino_y, self.dino_vy, action, self.speed if self.chrome_like else None
        )
        self._move_world()
        self._spawn_obstacles()

        self.ticks += 1
        self.distance += self.speed
        self.speed = min(MAX_SPEED, self.speed + SPEED_RAMP)
        self.score += self.speed * 0.025

        box = self.dino_box()
        for o in self.obstacles:
            if _overlap(box, _hitbox(o.x, o.y, o.w, o.h)):
                self.done = True
                break
        return not self.done

    # ------------------------------------------------------------------
    def _move_world(self):
        move = self.speed * self.motion_scale
        for o in self.obstacles:
            o.x -= move
        before = len(self.obstacles)
        self.obstacles = [o for o in self.obstacles if o.x + o.w > 0]
        # an obstacle that scrolled off the left edge was dodged
        self.passed += before - len(self.obstacles)
        self._next_spawn_x -= move

    def _spawn_obstacles(self):
        if self._next_spawn_x > WORLD_WIDTH:
            return
        if self.chrome_like:
            self._spawn_chrome_like()
            return
        kind = self._pick_kind()
        w, h = OBSTACLE_SIZES[kind]
        y = self.rng.choice(PTERO_HEIGHTS) if kind == PTERO else 0
        self.obstacles.append(Obstacle(kind, WORLD_WIDTH, y, w, h))
        min_gap = MIN_GAP_TICKS * self.speed
        gap = self.rng.uniform(min_gap, min_gap * MAX_GAP_FACTOR)
        self._next_spawn_x = WORLD_WIDTH + w + gap

    def _pick_kind(self) -> str:
        kinds = [CACTUS_SMALL, CACTUS_LARGE]
        if self.score >= PTERO_MIN_SCORE:
            kinds.append(PTERO)
        return self.rng.choice(kinds)

    def _spawn_chrome_like(self):
        kinds = [k for k in CHROME_SIZES if self.speed >= CHROME_MIN_SPEED[k]]
        recent = self._recent[-MAX_DUPLICATES:]
        if len(recent) == MAX_DUPLICATES and len(set(recent)) == 1 and len(kinds) > 1:
            kinds = [k for k in kinds if k != recent[0]]   # no third identical type in a row
        kind = self.rng.choice(kinds)
        self._recent.append(kind)

        size = self.rng.randint(1, MAX_CLUSTER)
        if size > 1 and self.speed < CHROME_MULTIPLE_SPEED[kind]:
            size = 1
        unit_w, h = CHROME_SIZES[kind]
        w = unit_w * size
        y = self.rng.choice(CHROME_PTERO_HEIGHTS) if kind == PTERO else 0
        self.obstacles.append(Obstacle(kind, WORLD_WIDTH, y, w, h, size))

        min_gap = round(w * self.speed + CHROME_MIN_GAP[kind] * GAP_COEFFICIENT)
        gap = self.rng.uniform(min_gap, min_gap * MAX_GAP_COEFFICIENT)
        self._next_spawn_x = WORLD_WIDTH + w + gap
