"""Is every game winnable? (Can a perfect player survive any seed?)

Obstacles never react to the dino, so for a given seed the whole obstacle course
is fixed. That lets us do something stronger than "my bot survived":

1. Record the course with the dino made invincible (obstacle positions per tick).
2. Track the SET of every dino state (height, velocity) that is still alive.
   At each tick, try all 3 actions from every state and keep the ones that
   don't hit anything.
3. If that set ever becomes empty, the dino was doomed - no sequence of actions
   could have survived. The game would be unfair.
"""

import pytest

from trex.game import DUCK, JUMP, NOOP, TRexGame, _hitbox, _overlap, dino_hitbox, dino_physics


def record_course(seed: int, ticks: int):
    """Obstacle hitboxes after every tick, with collisions ignored."""
    g, course = TRexGame(seed), []
    for _ in range(ticks):
        g.step(NOOP)
        g.done = False
        course.append([_hitbox(o.x, o.y, o.w, o.h) for o in g.obstacles])
    return course


# Exact search keeps ~5,000 live states per tick (slow). Instead, states that are within
# ~2 px of height / ~1 px/tick of velocity share one slot, and we keep one REAL
# representative for each. This is safe for proving solvability: every state we keep is
# genuinely reachable, so we can miss solutions but never invent one. If a seed ever
# fails here, re-check it with the exact search (BUCKET = (0.001, 0.001)) before blaming the game.
BUCKET = (2.0, 1.0)


def alive_states(course):
    """Yield (tick, number of live dino states). Stops early if none are left."""
    states = {(0.0, 0.0): (0.0, 0.0)}
    for t, boxes in enumerate(course):
        nxt = {}
        for y, vy in states.values():
            for action in (NOOP, JUMP, DUCK):
                ny, nvy, ducking = dino_physics(y, vy, action)
                dino = dino_hitbox(ny, ducking)
                if not any(_overlap(dino, b) for b in boxes):
                    nxt[(round(ny / BUCKET[0]), round(nvy / BUCKET[1]))] = (ny, nvy)
        states = nxt
        yield t, len(states)
        if not states:
            return


@pytest.mark.parametrize("seed", range(10))
def test_every_seed_is_survivable_for_the_whole_early_and_mid_game(seed):
    course = record_course(seed, ticks=4000)
    last_tick, last_count = max(alive_states(course), key=lambda r: r[0])
    assert last_tick == len(course) - 1 and last_count > 0, (
        f"seed {seed}: no survivable action sequence past tick {last_tick}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_survivable_at_maximum_speed_too(seed):
    # max speed (13) is reached around tick 7000; check well beyond that
    course = record_course(seed, ticks=9000)
    last_tick, last_count = max(alive_states(course), key=lambda r: r[0])
    assert last_tick == len(course) - 1 and last_count > 0, (
        f"seed {seed}: no survivable action sequence past tick {last_tick}"
    )
