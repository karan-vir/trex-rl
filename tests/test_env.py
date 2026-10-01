"""The Gymnasium wrapper: does it follow the API, and does it behave as designed?"""

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import trex  # noqa: F401  (registers "TRex-v0")
from trex.env import OBS_NAMES, TRexEnv
from trex.game import DINO_X, JUMP, NOOP


def test_passes_gymnasiums_official_checker():
    check_env(TRexEnv(), skip_render_check=True)


def test_can_be_created_by_name():
    env = gym.make("TRex-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (len(OBS_NAMES),)


def test_observations_always_fit_the_declared_space():
    env = TRexEnv()
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    for t in range(2000):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            env.reset()


def test_same_seed_gives_identical_episode():
    def episode(seed):
        env = TRexEnv()
        obs, _ = env.reset(seed=seed)
        out = [obs.copy()]
        for t in range(300):
            obs, r, term, trunc, _ = env.step(JUMP if t % 40 < 2 else NOOP)
            out.append(obs.copy())
            if term or trunc:
                break
        return np.array(out)
    assert np.array_equal(episode(3), episode(3))
    assert not np.array_equal(episode(3), episode(4))


def test_reward_is_survive_per_tick_then_death_penalty():
    env = TRexEnv(survive_reward=1.0, death_reward=-100.0)
    env.reset(seed=0)
    rewards = []
    while True:
        _, r, terminated, truncated, _ = env.step(NOOP)   # never jumps: dies at first obstacle
        rewards.append(r)
        if terminated:
            break
    assert terminated and not truncated
    assert rewards[-1] == -100.0
    assert all(r == 1.0 for r in rewards[:-1])


def test_pass_reward_is_paid_when_an_obstacle_is_dodged():
    env = TRexEnv(pass_reward=10.0)
    env.reset(seed=0)
    # make the course trivial: one cactus that scrolls past a dino that jumps over it
    # (cheat by making the dino invincible-and-idle in the underlying game)
    total_bonus = 0.0
    for _ in range(600):
        _, r, *_ = env.step(NOOP)
        env.game.done = False             # ignore crashes: we only want to count dodges
        total_bonus += r - 1.0 if r > 0 else 0.0
    assert env.game.passed > 0
    assert total_bonus == 10.0 * env.game.passed


def test_long_survival_is_truncated_not_terminated():
    env = TRexEnv(max_episode_steps=50)
    env.reset(seed=0)
    for t in range(50):
        _, _, terminated, truncated, _ = env.step(NOOP)
        env.game.done = False             # stay alive artificially
        if truncated:
            break
    assert truncated and not terminated
    assert t == 49


def test_obstacle_features_reflect_the_game():
    env = TRexEnv()
    env.reset(seed=0)
    for _ in range(60):
        obs, *_ = env.step(NOOP)
        env.game.done = False
    first = env.game.next_obstacles(1)[0]
    i = OBS_NAMES.index("obs0_dist")
    expected = (first.x - (DINO_X + 44)) / 600
    assert abs(obs[i] - expected) < 1e-4


def test_no_obstacle_looks_far_away_and_empty():
    env = TRexEnv()
    obs, _ = env.reset(seed=0)           # nothing has spawned yet
    assert obs[OBS_NAMES.index("obs0_dist")] > 1.0
    assert obs[OBS_NAMES.index("obs0_width")] == 0.0


def test_rgb_array_rendering_returns_a_frame():
    env = TRexEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (200, 600, 3) and frame.dtype == np.uint8
    env.close()
