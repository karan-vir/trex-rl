"""Studio backend: knob catalogue, state listing, and the server-side watch session."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from trex.studio import server


def test_knobs_cover_live_settings():
    k = server.knobs()
    keys = {x["key"] for x in k["knobs"]}
    assert server.LIVE <= keys
    assert all(x["options"] and x["help"] for x in k["knobs"])


def test_state_lists_runs_and_checkpoints():
    s = server.state()
    assert {"runs", "evals", "ckpts"} <= set(s)


def test_watch_rule_and_human():
    w = server.Watch()
    snap = w.new({"agent": "rule", "seed": 1})
    assert snap["png"] and snap["kind"] == "rule"
    snap = w.step(30, 0, False)
    assert snap["decisions"] == 30
    w.new({"agent": "human", "seed": 2})
    assert w.step(10, 1, True)["decisions"] == 10
