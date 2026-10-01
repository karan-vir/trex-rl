"""chrome_like mode: the real game's obstacle generator, jump, and speed quirk."""

import pytest

from trex.agents.rule_based import RuleBasedAgent
from trex.env import OBS_NAMES, TRexEnv
from trex.game import (
    CHROME_MIN_GAP, CHROME_MIN_SPEED, CHROME_MULTIPLE_SPEED, CHROME_PTERO_HEIGHTS, CLEAR_TICKS,
    DINO_X, GAP_COEFFICIENT, JUMP, MAX_DUPLICATES, NOOP, PTERO, START_SPEED, WORLD_WIDTH,
    CACTUS_LARGE, CACTUS_SMALL, Obstacle, TRexGame, _hitbox, _overlap, dino_hitbox, dino_physics,
)

KINDS = (CACTUS_SMALL, CACTUS_LARGE, PTERO)


def spawn_log(seed, ticks=9000, **kw):
    """Every obstacle as it appears: (kind, size, width, y, speed_at_spawn, tick)."""
    g, seen, log = TRexGame(seed, chrome_like=True, **kw), set(), []
    for t in range(ticks):
        g.step(NOOP)
        g.done = False
        for o in g.obstacles:
            if id(o) not in seen:
                seen.add(id(o))
                log.append((o, g.speed, t))
    return log


def test_groups_only_appear_above_their_speed_threshold():
    sizes = set()
    for seed in range(15):
        for o, speed, _ in spawn_log(seed):
            sizes.add(o.size)
            if o.size > 1:
                assert speed >= CHROME_MULTIPLE_SPEED[o.kind] - 0.01
    assert sizes == {1, 2, 3}


def test_pterodactyls_wait_for_speed_8_5_and_use_the_three_heights():
    heights = set()
    for seed in range(15):
        for o, speed, _ in spawn_log(seed):
            if o.kind == PTERO:
                assert speed >= CHROME_MIN_SPEED[PTERO] - 0.01
                heights.add(o.y)
                assert o.size == 1
    assert heights == set(CHROME_PTERO_HEIGHTS)


def test_gap_after_an_obstacle_grows_with_its_width_and_the_speed():
    for seed in range(8):
        # measure the real gap when the next one spawns: it starts at x = WORLD_WIDTH
        g = TRexGame(seed, chrome_like=True)
        for _ in range(7000):
            g.step(NOOP)
            g.done = False
            if len(g.obstacles) >= 2 and g.obstacles[-1].x >= WORLD_WIDTH - g.speed - 1:
                a, b = g.obstacles[-2], g.obstacles[-1]
                gap = b.x - (a.x + a.w)
                lo = a.w * (g.speed - 0.1) + CHROME_MIN_GAP[a.kind] * GAP_COEFFICIENT
                assert gap >= lo * 0.97, (a.kind, a.size, gap, lo)


def test_never_more_than_two_identical_kinds_in_a_row():
    for seed in range(15):
        kinds = [o.kind for o, _, _ in spawn_log(seed)]
        for i in range(len(kinds) - MAX_DUPLICATES):
            assert len(set(kinds[i:i + MAX_DUPLICATES + 1])) > 1


def test_first_obstacle_arrives_after_the_clear_time():
    first_tick = spawn_log(0, ticks=1000)[0][2]
    assert first_tick >= CLEAR_TICKS - 5


# --- the real jump -------------------------------------------------------------
def jump(speed):
    y, vy, _ = dino_physics(0.0, 0.0, JUMP, speed)
    ys = [y]
    while y > 0:
        y, vy, _ = dino_physics(y, vy, NOOP, speed)
        ys.append(y)
    return ys


def test_real_jump_apex_is_capped_so_speed_barely_changes_the_height():
    apexes = [max(jump(s)) for s in (6, 9, 13)]
    assert all(80 < a < 92 for a in apexes), apexes
    assert max(apexes) - min(apexes) < 6


def test_real_jump_stays_above_a_large_cactus_for_about_twenty_ticks():
    ys = jump(8.0)
    above = [i for i, y in enumerate(ys) if y > 50]
    assert 18 <= len(above) <= 24, len(above)      # measured on chromedino.com: ~22


# --- the speed quirk -----------------------------------------------------------
def test_motion_scale_slows_the_obstacles_but_not_the_reported_speed():
    def after(scale):
        g = TRexGame(0, chrome_like=True, motion_scale=scale)
        g.obstacles = [Obstacle(CACTUS_SMALL, 500.0, 0, 17, 35)]
        g._next_spawn_x = 10**6
        g.step(NOOP)
        return g.obstacles[0].x, g.speed
    (x_full, s_full), (x_slow, s_slow) = after(1.0), after(0.87)
    assert s_full == s_slow                          # observed speed unchanged
    assert 500 - x_slow == pytest.approx(0.87 * (500 - x_full))


# --- the width-aware rule ---------------------------------------------------------
def obs_for(width_px, gap_px=60, speed=8.0):
    env = TRexEnv(chrome_like=True)
    env.reset(seed=0)
    env.game.speed = speed
    env.game._next_spawn_x = 10**6
    env.game.obstacles = [Obstacle(CACTUS_LARGE, DINO_X + 44 + gap_px, 0, width_px, 50)]
    return env._observe()


def test_width_aware_rule_jumps_later_for_wider_obstacles():
    agent = RuleBasedAgent(width_aware=True, arc_center=16.5)
    # 100 px away at speed 8 (12.5 ticks): a single cactus is jumped now, a 3-group waits
    assert agent.act(obs_for(25, gap_px=100)) == JUMP
    assert agent.act(obs_for(75, gap_px=100)) == NOOP
    assert agent.act(obs_for(75, gap_px=60)) == JUMP     # ...until it is closer


def test_speed_scale_makes_the_agent_expect_a_later_arrival():
    plain = RuleBasedAgent(12.0)
    scaled = RuleBasedAgent(12.0, speed_scale=0.87)
    o = obs_for(25, gap_px=90, speed=8.0)         # 11.25 ticks at 8 px/tick, 12.9 at 6.96
    assert plain.act(o) == JUMP and scaled.act(o) == NOOP


# --- fairness: can a perfect player survive the real generator? --------------------
def course_with_speeds(seed, ticks):
    g, rows = TRexGame(seed, chrome_like=True), []
    for _ in range(ticks):
        speed = g.speed
        g.step(NOOP)
        g.done = False
        rows.append((speed, [_hitbox(o.x, o.y, o.w, o.h) for o in g.obstacles]))
    return rows


def survivable(rows):
    states = {(0, 0): (0.0, 0.0)}
    for speed, boxes in rows:
        nxt = {}
        for y, vy in states.values():
            for action in (0, 1, 2):
                ny, nvy, ducking = dino_physics(y, vy, action, speed)
                box = dino_hitbox(ny, ducking)
                if not any(_overlap(box, b) for b in boxes):
                    nxt[(round(ny / 2.0), round(nvy / 1.0))] = (ny, nvy)
        states = nxt
        if not states:
            return False
    return True


@pytest.mark.parametrize("seed", range(6))
def test_perfect_player_can_survive_the_real_generator(seed):
    assert survivable(course_with_speeds(seed, 6000))
