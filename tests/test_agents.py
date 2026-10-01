"""Agents: do the hand-written rules make the decisions we expect?"""

import numpy as np

from trex.agents.random_agent import RandomAgent
from trex.agents.rule_based import RuleBasedAgent
from trex.env import TRexEnv
from trex.evaluate import evaluate
from trex.game import CACTUS_SMALL, DINO_X, DUCK, JUMP, NOOP, PTERO, OBSTACLE_SIZES, Obstacle


def observe_with(kind, gap_px, y=0.0, speed_ticks=0):
    """Observation for a game with ONE obstacle `gap_px` in front of the dino's nose."""
    env = TRexEnv()
    env.reset(seed=0)
    env.game._next_spawn_x = 10**6
    for _ in range(speed_ticks):
        env.game.step(NOOP)
    w, h = OBSTACLE_SIZES[kind]
    env.game.obstacles = [Obstacle(kind, DINO_X + 44 + gap_px, y, w, h)]
    return env._observe()


def test_does_nothing_when_nothing_is_ahead():
    env = TRexEnv(); obs, _ = env.reset(seed=0)
    assert RuleBasedAgent().act(obs) == NOOP


def test_waits_while_a_cactus_is_far_then_jumps_when_close():
    agent = RuleBasedAgent(lead_ticks=12)
    assert agent.act(observe_with(CACTUS_SMALL, gap_px=400)) == NOOP
    assert agent.act(observe_with(CACTUS_SMALL, gap_px=40)) == JUMP     # 40px / 6 = ~7 ticks


def test_trigger_distance_grows_with_speed():
    agent = RuleBasedAgent(lead_ticks=12)
    gap = 100    # 100px: ~17 ticks away at speed 6, but only ~8 ticks at speed 13
    slow = observe_with(CACTUS_SMALL, gap)
    fast_env = TRexEnv(); fast_env.reset(seed=0)
    fast_env.game.speed = 13.0
    fast_env.game._next_spawn_x = 10**6
    fast_env.game.obstacles = [Obstacle(CACTUS_SMALL, DINO_X + 44 + gap, 0, 17, 35)]
    fast = fast_env._observe()
    assert agent.act(slow) == NOOP and agent.act(fast) == JUMP


def test_ducks_under_a_mid_pterodactyl_and_ignores_a_high_one():
    agent = RuleBasedAgent()
    assert agent.act(observe_with(PTERO, 40, y=30)) == DUCK
    assert agent.act(observe_with(PTERO, 40, y=60)) == NOOP


def test_jumps_over_a_low_pterodactyl():
    assert RuleBasedAgent().act(observe_with(PTERO, 40, y=8)) == JUMP


def test_does_not_try_to_jump_again_in_mid_air():
    agent = RuleBasedAgent()
    env = TRexEnv(); env.reset(seed=0)
    env.game.dino_y = 30.0; env.game.dino_vy = 2.0
    env.game._next_spawn_x = 10**6
    env.game.obstacles = [Obstacle(CACTUS_SMALL, DINO_X + 44 + 40, 0, 17, 35)]
    assert agent.act(env._observe()) == NOOP


def test_rule_based_agent_survives_every_fixed_seed_for_a_long_time():
    result = evaluate(RuleBasedAgent(), seeds=range(500, 520), max_episode_steps=5_000)
    assert result.survived == 20, result.deaths


def test_random_agent_is_far_worse_and_evaluation_is_repeatable():
    a = evaluate(RandomAgent(seed=1), seeds=range(30))
    b = evaluate(RandomAgent(seed=1), seeds=range(30))
    assert a.scores == b.scores
    assert a.survived == 0 and np.mean(a.scores) < 1000
