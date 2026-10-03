"""Export a trained PPO policy to a single JavaScript file that plays the real game in the browser.

    python scripts/export_policy_js.py runs/ppo_b/best --out scripts/chrome_dino_policy.js

The file contains the network weights, the code that builds the same 22-number observation as
trex/real_env.py from the game's own objects, and a self-test: test vectors computed here (the
network's logits on random observations, and observations built from real engine states) that the
browser re-computes on load and compares.
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from trex.engine import PTERODACTYL
from trex.real_env import RealTRexEnv


def runner_state(env):
    e = env.engine
    return {
        "tRex": {"yPos": e.y, "jumpVelocity": e.vel, "jumping": e.jumping, "ducking": e.ducking,
                 "speedDrop": e.speed_drop, "status": e.status},
        "currentSpeed": e.speed,
        "horizon": {"obstacles": [
            {"xPos": o.x, "yPos": o.y, "width": o.width, "size": o.size,
             "typeConfig": {"height": o.height, "type": "PTERODACTYL" if o.kind == PTERODACTYL else "CACTUS"}}
            for o in e.obstacles]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", default="scripts/chrome_dino_policy.js")
    ap.add_argument("--version", default=None)
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    pol = model.policy
    mlp = pol.mlp_extractor.policy_net
    arrs = [mlp[0].weight, mlp[0].bias, mlp[2].weight, mlp[2].bias, pol.action_net.weight, pol.action_net.bias]
    arrs = [a.detach().numpy().astype(np.float64) for a in arrs]
    h, n_in = arrs[0].shape
    scales, q = [], []
    for a in arrs:                                     # int16 quantisation, one scale per tensor
        sc = float(np.abs(a).max()) / 32767.0
        scales.append(sc)
        q.append(np.round(a / sc).astype(np.int16).ravel())
    blob = np.concatenate(q).astype("<i2").tobytes()
    policy = {"n_in": int(n_in), "h": int(h), "scales": scales, "q": base64.b64encode(blob).decode()}
    # check the quantised network picks the same actions as the original on real observations
    deq = [np.round(a / sc) * sc for a, sc in zip(arrs, scales)]
    def fwd(x, P):
        h1 = np.tanh(x @ P[0].T + P[1]); h2 = np.tanh(h1 @ P[2].T + P[3]); return h2 @ P[4].T + P[5]
    probe_env = RealTRexEnv(max_frames=3000, randomize=True)
    o, _ = probe_env.reset(seed=7); X = []
    for _ in range(5000):
        X.append(o.copy()); a, _ = model.predict(o, deterministic=True); o, _, te, tu, _ = probe_env.step(int(a))
        if te or tu: o, _ = probe_env.reset()
    X = np.array(X, dtype=np.float64)
    agree = float((np.argmax(fwd(X, arrs), 1) == np.argmax(fwd(X, deq), 1)).mean())
    print(f"int16 quantisation: same action on {agree:.4%} of {len(X)} observations, max logit diff {np.abs(fwd(X, arrs)-fwd(X, deq)).max():.4f}")

    # test vectors: (a) network logits on observations drawn from real play, (b) observation building
    env = RealTRexEnv(max_frames=4000, randomize=True)
    obs, _ = env.reset(seed=123)
    rng = np.random.default_rng(0)
    logits_cases, state_cases, seen = [], [], 0
    for step in range(6000):
        if step % 97 == 0 and len(state_cases) < 12:
            state_cases.append({"runner": runner_state(env), "eff": float(env.eff_speed), "dt": float(env.last_elapsed),
                                "held": int(env.held), "obs": [round(float(x), 7) for x in obs]})
        if step % 53 == 0 and len(logits_cases) < 12:
            with torch.no_grad():
                lg = pol.get_distribution(torch.as_tensor(obs[None])).distribution.logits[0].numpy()
            logits_cases.append({"obs": [round(float(x), 7) for x in obs], "logits": [round(float(x), 6) for x in lg]})
        act, _ = model.predict(obs, deterministic=False)
        obs, rew, te, tu, info = env.step(int(act))
        if te or tu:
            obs, _ = env.reset()
    selftest = {"logits": logits_cases, "states": state_cases}

    name = Path(args.model).parent.name + "/" + Path(args.model).stem
    version = args.version or f"ppo:{name}"
    src = Path("scripts/chrome_dino_policy.template.js").read_text()
    src = (src.replace("__POLICY_NAME__", name).replace("__VERSION__", version)
              .replace("__POLICY_JSON__", json.dumps(policy, separators=(",", ":")))
              .replace("__SELFTEST_JSON__", json.dumps(selftest, separators=(",", ":"))))
    Path(args.out).write_text(src)
    print(f"wrote {args.out}: {len(src)/1024:.0f} KB, network {n_in}-{h}-{h}-3, "
          f"{len(logits_cases)} logit cases, {len(state_cases)} observation cases, version {version}")


if __name__ == "__main__":
    main()
