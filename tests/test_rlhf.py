"""RLHF pipeline pieces: clips replay deterministically, the judge is consistent, the reward model learns."""

import numpy as np
import pytest
from stable_baselines3 import PPO

from trex.real_env import RealTRexEnv
from trex.rlhf import core
from trex.rlhf.reward_model import RewardNet, step_rewards, train_reward_model


@pytest.fixture(scope="module")
def model():
    return PPO("MlpPolicy", RealTRexEnv(), device="cpu", seed=0)


def test_judge_orders_by_survival_then_progress():
    a = dict(crashed=True, steps=40, passes=0)
    b = dict(crashed=False, steps=150, passes=2)
    assert core.judge(a, b) == "B" and core.judge(b, a) == "A"
    assert core.judge(a, dict(a, steps=45)) == "same"                    # too close to call
    assert core.judge(b, dict(b, passes=3)) == "B"
    assert core.judge(b, dict(b)) == "same"


def test_replay_is_deterministic(model):
    rng = np.random.default_rng(0)
    spec = core.new_spec(rng)
    acts = core.rollout(model, spec, 0.05, rng)
    r1, r2 = core.replay(spec, acts), core.replay(spec, acts)
    assert r1["crashed"] == r2["crashed"] and r1["steps"] == r2["steps"] == len(acts)
    assert np.array_equal(r1["X"], r2["X"])
    assert r1["X"].shape[1] == 25


def test_pair_records_a_verdict(model, tmp_path):
    rng = np.random.default_rng(1)
    pair = core.make_pair(model, rng)
    assert pair["truth"] in ("A", "B", "same") and pair["A"] and pair["B"]
    core.add_pref(pair, "A", "human", tmp_path)
    s = core.prefs_summary(tmp_path)
    assert s["total"] == 1 and s["human"] == 1 and s["a"] == 1


def test_render_frames(model):
    rng = np.random.default_rng(2)
    spec = core.new_spec(rng)
    acts = core.rollout(model, spec, 0.0, rng, T=30)
    r = core.replay(spec, acts, render=True, every=3)
    assert len(r["frames"]) >= len(acts) // 3


def test_reward_model_learns_from_oracle_labels(model):
    rng = np.random.default_rng(3)
    recs = []
    while len(recs) < 120:
        p = core.make_pair(model, rng)
        recs.append({"spec": p["spec"], "A": p["A"], "B": p["B"], "label": p["truth"]})
    rm, hist, info = train_reward_model(recs, epochs=25)
    assert hist[-1]["loss"] < hist[0]["loss"]
    assert isinstance(rm, RewardNet) and len(step_rewards(rm, np.zeros((4, 25), np.float32))) == 4
