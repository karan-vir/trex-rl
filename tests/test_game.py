"""Rules of the game: physics, collisions, spawning, difficulty, repeatability."""

import pytest

from trex.game import (
    DINO_X, DUCK, DUCK_H, DINO_H, JUMP, MAX_SPEED, MIN_GAP_TICKS, NOOP, PTERO,
    PTERO_HEIGHTS, PTERO_MIN_SCORE, START_SPEED, WORLD_WIDTH, Obstacle, TRexGame,
    dino_hitbox, dino_physics, OBSTACLE_SIZES, CACTUS_SMALL,
)


def scripted(t: int) -> int:
    """A fixed, silly action pattern - the point is that it is the SAME every time."""
    return JUMP if t % 41 < 2 else DUCK if t % 67 < 9 else NOOP


def run(seed: int, ticks: int, policy=scripted):
    g = TRexGame(seed)
    trace = []
    for t in range(ticks):
        g.step(policy(t))
        trace.append((g.dino_y, g.score, g.speed, [(o.kind, o.x, o.y) for o in g.obstacles]))
    return g, trace


# --- Repeatability -----------------------------------------------------------
def test_same_seed_same_actions_gives_identical_game():
    _, a = run(seed=5, ticks=1500, policy=lambda t: NOOP)
    _, b = run(seed=5, ticks=1500, policy=lambda t: NOOP)
    assert a == b


def test_different_seeds_give_different_obstacle_courses():
    # NOOP only -> the dino dies at the first obstacle, so compare the spawn order
    # by forcing the game to keep running.
    def courses(seed):
        g, kinds = TRexGame(seed), []
        for _ in range(3000):
            g.step(NOOP)
            g.done = False
            kinds += [(o.kind, round(o.x)) for o in g.obstacles if o.x > WORLD_WIDTH - 15]
        return kinds
    assert courses(1) != courses(2)


def test_reset_with_seed_replays_the_same_game():
    g = TRexGame(9)
    first = [g.step(NOOP) or (g.score, len(g.obstacles)) for _ in range(100)]
    g.reset(9)
    second = [g.step(NOOP) or (g.score, len(g.obstacles)) for _ in range(100)]
    assert first == second


# --- Dino physics ------------------------------------------------------------
def jump_trace(extra_action=NOOP):
    y, vy, _ = dino_physics(0.0, 0.0, JUMP)
    ys = [y]
    while y > 0:
        y, vy, _ = dino_physics(y, vy, extra_action)
        ys.append(y)
    return ys


def test_jump_has_sensible_height_and_airtime():
    ys = jump_trace()
    assert 75 < max(ys) < 90          # about 83 px: clears the 50 px large cactus
    assert 30 <= len(ys) <= 36        # about half a second at 60 ticks/s
    assert ys[-1] == 0                # and it comes back down


def test_cannot_jump_in_mid_air():
    y, vy, _ = dino_physics(0.0, 0.0, JUMP)
    y, vy, _ = dino_physics(y, vy, NOOP)
    y2, vy2, _ = dino_physics(y, vy, JUMP)       # ignored: not on the ground
    y3, vy3, _ = dino_physics(y, vy, NOOP)
    assert (y2, vy2) == (y3, vy3)


def test_holding_duck_in_mid_air_makes_you_fall_faster():
    assert len(jump_trace(DUCK)) < len(jump_trace(NOOP))


def test_ducking_only_on_the_ground_and_shrinks_the_hitbox():
    _, _, ducking = dino_physics(0.0, 0.0, DUCK)
    assert ducking
    _, _, ducking_air = dino_physics(30.0, 2.0, DUCK)
    assert not ducking_air
    stand_top = dino_hitbox(0, False)[3]
    duck_top = dino_hitbox(0, True)[3]
    assert duck_top < stand_top


def test_standing_still_on_the_ground_stays_put():
    assert dino_physics(0.0, 0.0, NOOP) == (0.0, 0.0, False)


# --- Collisions --------------------------------------------------------------
def place(g: TRexGame, kind: str, x: float, y: float = 0.0):
    w, h = OBSTACLE_SIZES[kind]
    g.obstacles = [Obstacle(kind, x, y, w, h)]
    g._next_spawn_x = 10**6      # keep other obstacles away


def test_hitting_an_obstacle_ends_the_game():
    g = TRexGame(0)
    place(g, CACTUS_SMALL, DINO_X + 10)
    alive = g.step(NOOP)
    assert not alive and g.done


def test_a_near_miss_inside_the_shrunken_hitbox_is_safe():
    g = TRexGame(0)
    # sprite edges would overlap by 2 px, but hitboxes are shrunk by 4 px per side
    place(g, CACTUS_SMALL, DINO_X + 44 - 2 + g.speed)   # will move left by speed this tick
    assert g.step(NOOP)


def test_jumping_clears_a_cactus_but_standing_does_not():
    def attempt(first_action):
        g = TRexGame(0)
        place(g, CACTUS_SMALL, 200)
        for t in range(120):
            g.obstacles[0:1] = g.obstacles[0:1]   # (no-op, keeps intent obvious)
            g._next_spawn_x = 10**6
            if not g.step(first_action if t == 15 else NOOP):
                return False
        return True
    assert attempt(JUMP)
    assert not attempt(NOOP)


def test_ducking_passes_under_a_mid_height_ptero_but_standing_does_not():
    def attempt(action):
        g = TRexGame(0)
        place(g, PTERO, 300, y=30)
        for _ in range(80):
            g._next_spawn_x = 10**6
            if not g.step(action):
                return False
        return True
    assert attempt(DUCK)
    assert not attempt(NOOP)


def test_a_high_ptero_flies_over_a_standing_dino():
    g = TRexGame(0)
    place(g, PTERO, 300, y=60)
    for _ in range(80):
        g._next_spawn_x = 10**6
        assert g.step(NOOP)


def test_step_after_game_over_changes_nothing():
    g = TRexGame(0)
    place(g, CACTUS_SMALL, DINO_X + 10)
    g.step(NOOP)
    snapshot = (g.score, g.ticks, g.dino_y, [o.x for o in g.obstacles])
    assert g.step(JUMP) is False
    assert snapshot == (g.score, g.ticks, g.dino_y, [o.x for o in g.obstacles])


# --- Spawning and difficulty ---------------------------------------------------
def survive(seed: int, ticks: int):
    """Run with collisions ignored, yielding the game after every tick."""
    g = TRexGame(seed)
    for _ in range(ticks):
        g.step(NOOP)
        g.done = False
        yield g


def test_obstacles_appear_off_screen_on_the_right():
    seen_new = []
    prev = 0
    for g in survive(3, 3000):
        if len(g.obstacles) > prev or (g.obstacles and g.obstacles[-1].x > WORLD_WIDTH - 15):
            seen_new.append(g.obstacles[-1].x)
        prev = len(g.obstacles)
    assert seen_new and all(x >= WORLD_WIDTH - 15 for x in seen_new)


def test_no_pterodactyls_in_the_early_game():
    for g in survive(4, 3000):
        if g.score < PTERO_MIN_SCORE:
            assert all(o.kind != PTERO for o in g.obstacles)


def test_pterodactyls_do_appear_later_and_use_the_three_heights():
    heights = set()
    for g in survive(4, 6000):
        heights |= {o.y for o in g.obstacles if o.kind == PTERO}
    assert heights == set(PTERO_HEIGHTS)


def test_gap_between_obstacles_is_at_least_a_jump_long():
    for seed in range(5):
        for g in survive(seed, 4000):
            for left, right in zip(g.obstacles, g.obstacles[1:]):
                gap = right.x - (left.x + left.w)
                # speed creeps up a little between spawn and now, hence the small slack
                assert gap >= (MIN_GAP_TICKS - 2) * START_SPEED


def test_speed_ramps_up_and_is_capped():
    g = TRexGame(0)
    assert g.speed == START_SPEED
    for _ in range(100):
        g.step(NOOP); g.done = False
    assert g.speed > START_SPEED
    for _ in range(10_000):
        g.step(NOOP); g.done = False
    assert g.speed == MAX_SPEED


def test_score_only_goes_up():
    last = -1.0
    for g in survive(1, 500):
        assert g.score > last
        last = g.score
