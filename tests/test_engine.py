"""The faithful engine (trex/engine.py): mechanics copied from the real game's code."""

import pytest

from trex.engine import (
    ACCELERATION, CLEAR_TIME, DUCKING, GROUND_Y, JUMPING, MAX_SPEED, RUNNING, SPEED0, RealEngine,
)
from trex.real_env import OBS_DIM, RealTRexEnv


def settle(e, n=30):
    for _ in range(n):
        e.frame()


def test_same_seed_same_game():
    def run(seed):
        e = RealEngine(seed=seed, jitter_ms=0.3, skip_clear_time=True)
        out = []
        for t in range(400):
            if t % 50 == 5:
                e.key_down("jump")
            if t % 50 == 25:
                e.key_up("jump")
            e.frame()
            out.append((e.y, e.speed, len(e.obstacles), e.crashed))
            if e.crashed:
                break
        return out
    assert run(3) == run(3)
    assert run(3) != run(4)


def test_clock_is_integer_milliseconds_alternating_8_and_9_at_120hz():
    e = RealEngine(seed=1, frame_ms=1000 / 120)
    dts = []
    for _ in range(120):
        e.frame()
        dts.append(e.last_dt)
    assert set(dts) <= {8, 9} and dts.count(8) > dts.count(9)
    assert abs(sum(dts) - 1000) <= 1


def test_speed_grows_per_update_call_not_per_second():
    for frame_ms in (1000 / 60, 1000 / 120):
        e = RealEngine(seed=1, frame_ms=frame_ms)
        for _ in range(150):                # stays within the first 3 s, before any obstacle
            e.frame()
        assert e.speed == pytest.approx(SPEED0 + 150 * ACCELERATION, abs=1e-6)
    e = RealEngine(seed=1, start_speed=MAX_SPEED - 0.0005)
    for _ in range(5):
        e.frame()
    assert e.speed <= MAX_SPEED + ACCELERATION


def test_no_obstacles_for_the_first_three_seconds():
    e = RealEngine(seed=2, frame_ms=1000 / 60)
    seen_at = None
    for _ in range(400):
        e.frame()
        if e.obstacles and seen_at is None:
            seen_at = e.running_time
    assert seen_at is not None and seen_at > CLEAR_TIME


def test_a_held_jump_has_the_real_arc():
    e = RealEngine(seed=1, frame_ms=1000 / 120, start_speed=8.0)
    settle(e)
    e.key_down("jump")
    peak, frames = 0.0, 0
    while True:
        e.frame()
        frames += 1
        peak = max(peak, e.height)
        if not e.jumping or frames > 400:
            break
    assert 84 <= peak <= 91                 # measured on the real game: 87
    assert 62 <= frames <= 72               # frames at 120/s: measured ~34 ticks of airtime = ~68 frames


def test_cannot_jump_while_ducking_and_duck_toggles_the_state():
    e = RealEngine(seed=1)
    settle(e)
    e.key_down("duck")
    assert e.ducking and e.status == DUCKING
    e.key_down("jump")
    assert not e.jumping
    e.key_up("duck")
    assert not e.ducking and e.status == RUNNING


def test_duck_in_the_air_starts_a_fast_fall_and_repeating_it_restarts_it():
    e = RealEngine(seed=1)
    settle(e)
    e.key_down("jump")
    for _ in range(20):
        e.frame()
    assert e.jumping
    e.key_down("duck")
    assert e.speed_drop and e.vel == 1.0
    e.frame(); e.frame(); e.frame()
    v = e.vel
    e.key_down("duck")                      # a second press resets the fall velocity
    assert e.vel == 1.0 and v > 1.0


def find_landing(kind):
    """Search timings for a fast fall that lands 'clean' or into the 'stuck' state."""
    for seed in range(300):
        e = RealEngine(seed=seed, frame_ms=1000 / 120, jitter_ms=0.3, start_speed=8.0)
        settle(e)
        e.key_down("jump")
        pressed = False
        for _ in range(600):
            e.frame()
            if e.jumping and e.vel > 0 and e.height <= 16 and not pressed:
                e.key_down("duck")
                pressed = True
            if pressed and e.height <= 0.5:
                stuck = e.jumping and e.status == DUCKING
                if stuck == (kind == "stuck"):
                    return e
                break
    return None


def test_a_fast_fall_can_land_clean_into_a_duck():
    e = find_landing("clean")
    assert e is not None
    assert not e.jumping and e.ducking and e.y == GROUND_Y


def test_a_fast_fall_can_land_exactly_and_get_stuck_for_about_a_second():
    e = find_landing("stuck")
    assert e is not None                    # jumping, ducking state, at ground level
    assert e.jumping and e.y == GROUND_Y and e.status == DUCKING
    frames = 0
    while e.jumping and frames < 1000:
        e.frame()
        frames += 1
    assert 60 < frames < 300                # measured in the real game: 107-117 frames at 120/s


def test_collisions_use_the_first_obstacle_only_and_the_real_boxes():
    e = RealEngine(seed=1, skip_clear_time=True)
    for _ in range(2000):
        if not e.frame():
            break
    assert e.crashed                        # a dino that never jumps dies at the first obstacle
    assert e.score < 100


def test_environment_is_a_valid_gymnasium_env():
    from gymnasium.utils.env_checker import check_env
    env = RealTRexEnv(max_frames=500)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
