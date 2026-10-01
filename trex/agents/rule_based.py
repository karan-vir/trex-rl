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
    DINO_W, DUCK, HITBOX_SHRINK, JUMP, MAX_SPEED, NOOP, START_SPEED, WORLD_WIDTH,
)

_I = {name: i for i, name in enumerate(OBS_NAMES)}


class RuleBasedAgent:
    def __init__(self, lead_ticks: float = 12.0, width_aware: bool = False,
                 arc_center: float = 15.5, speed_scale: float = 1.0):
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

    def act(self, obs) -> int:
        if obs[_I["obs0_width"]] == 0:                 # nothing ahead
            return NOOP

        speed = (START_SPEED + obs[_I["speed"]] * (MAX_SPEED - START_SPEED)) * self.speed_scale  # px per tick
        dist = obs[_I["obs0_dist"]] * WORLD_WIDTH                            # px to the dino's nose
        fly_y = obs[_I["obs0_y"]] * FLY_SCALE                                # height off the ground
        on_ground = obs[_I["dino_y"]] <= 0
        arrives_in = dist / speed                                            # ticks until contact

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
