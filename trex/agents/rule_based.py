"""A hand-written player. No learning: a human decided the rules.

It sees exactly what a learning agent sees (the 12-number observation), so it is a
fair baseline. Writing it also shows what a learning agent must discover by itself:
WHEN to jump, and that the answer depends on speed.

Core idea: convert distance into time. "The cactus is 200 px away" means nothing
by itself; "it arrives in 14 ticks" does, because a jump takes about 33 ticks.
"""

from __future__ import annotations

from trex.env import FLY_SCALE, OBS_NAMES, SIZE_SCALE
from trex.game import (
    DINO_W, DUCK, GRAVITY, HITBOX_SHRINK, JUMP, MAX_SPEED, NOOP, START_SPEED, WORLD_WIDTH,
)
from trex.env import VY_SCALE, Y_SCALE

_I = {name: i for i, name in enumerate(OBS_NAMES)}


def ticks_to_land(y: float, vy: float, fast: bool) -> int:
    """Ticks until the dino touches down from height y with upward velocity vy.
    fast=True means a fast fall (duck pressed once): velocity 1 down, displacement tripled."""
    t = 0
    if fast:
        vy = -1.0
    while y > 0 and t < 80:
        y += vy * (3.0 if fast else 1.0)
        vy -= GRAVITY
        t += 1
    return t


class RuleBasedAgent:
    def __init__(self, lead_ticks: float = 12.0, width_aware: bool = False,
                 arc_center: float = 15.5, speed_scale: float = 1.0,
                 fast_fall: bool = False, min_lead_large: float = 6.5, min_lead_small: float = 5.0):
        # How many ticks before an obstacle arrives we react. Too late and we hit it on
        # the way up; too early and we land on it. Measured: 6..18 all survive every
        # seed, 4 and 22+ fail every seed. 12 is the middle of that window.
        self.lead_ticks = lead_ticks
        # Width-aware: a wide obstacle (a group of cacti) must be covered by the high part
        # of the jump for longer, so jump later. Aim the CENTER of the arc at the CENTER of
        # the time the dino overlaps the obstacle:
        #     lead = arc_center - (obstacle_width + dino_width) / (2 * speed)
        # arc_center = ticks after takeoff at the middle of the time spent above a cactus.
        self.width_aware = width_aware
        self.arc_center = arc_center
        # If obstacles really move at only a fraction of the reported speed (see
        # TRexGame.motion_scale), use that fraction. The Chrome script measures it live.
        self.speed_scale = speed_scale
        # Fast fall: once past an obstacle, if the NEXT one would arrive before a normal landing
        # leaves time to act, press duck (once) to drop to the ground sooner.
        self.fast_fall = fast_fall
        # Smallest warning (ticks before the obstacle arrives) with which a jump still clears it:
        # the jump has to be above the cactus top by the time the nose reaches it (measured: ~5.6 ticks for 50 px).
        self.min_lead_large, self.min_lead_small = min_lead_large, min_lead_small

    def act(self, obs) -> int:
        if obs[_I["obs0_width"]] == 0:                 # nothing ahead
            return NOOP

        speed = (START_SPEED + obs[_I["speed"]] * (MAX_SPEED - START_SPEED)) * self.speed_scale  # px per tick
        dist = obs[_I["obs0_dist"]] * WORLD_WIDTH                            # px to the dino's nose
        fly_y = obs[_I["obs0_y"]] * FLY_SCALE                                # height off the ground
        on_ground = obs[_I["dino_y"]] <= 0
        arrives_in = dist / speed                                            # ticks until contact

        if not on_ground and self.fast_fall and dist > 0:
            # Airborne, and the obstacle ahead is past the one we jumped over: if a normal landing
            # would leave less than the MINIMUM workable warning, but a fast fall would leave enough,
            # fall fast. (Real traces: the dino lands with the next group only ~25 px away.)
            y = obs[_I["dino_y"]] * Y_SCALE
            vy = obs[_I["dino_vy"]] * VY_SCALE
            if vy <= 0 and fly_y < 45:                       # descending; a high pterodactyl needs nothing
                height = obs[_I["obs0_height"]] * SIZE_SCALE
                need = 1.0 if fly_y >= 20 else (self.min_lead_large if height >= 45 else self.min_lead_small)
                slow, quick = ticks_to_land(y, vy, False), ticks_to_land(y, vy, True)
                if arrives_in - slow < need <= arrives_in - quick:
                    return DUCK
        if fly_y >= 45:                    # high pterodactyl: flies over a standing dino
            return NOOP
        if fly_y >= 20:                    # mid pterodactyl: duck underneath
            return DUCK if arrives_in < self.lead_ticks else NOOP
        # cactus or low pterodactyl: jump over it
        lead = self.lead_ticks
        if self.width_aware:
            width = obs[_I["obs0_width"]] * SIZE_SCALE
            lead = self.arc_center - (width + DINO_W - 2 * HITBOX_SHRINK) / (2 * speed)
        if on_ground and 0 <= arrives_in < lead:
            return JUMP
        return NOOP
