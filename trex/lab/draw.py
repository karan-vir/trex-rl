"""Drawing helpers for the lab: the game itself (from the faithful engine) and the "mind" panels."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import pygame

from trex import sprites
from trex.engine import (
    GROUND_Y, PTERODACTYL, T_START_X, TREX_BOXES_DUCKING, TREX_BOXES_RUNNING, WIDTH,
)
from trex.lab.session import ACTION_NAMES, OBS_NAMES, Session

# ---- colours -----------------------------------------------------------------------------
BG = (24, 25, 28)
PANEL = (34, 35, 39)
INK = (232, 231, 226)
MUTE = (150, 149, 143)
LINE = (62, 63, 68)
GAME_BG = (247, 247, 247)
BLUE, GREEN, ORANGE, PURPLE, RED = (110, 160, 255), (76, 195, 138), (240, 165, 74), (192, 140, 245), (240, 106, 106)

_rng = random.Random(1)
PEBBLES = [(_rng.randrange(WIDTH), _rng.randrange(6, 30), _rng.choice((2, 3, 5))) for _ in range(45)]
CLOUDS = [(40, 30), (230, 55), (420, 25), (560, 60)]
_scale_cache: dict = {}


def fonts():
    pygame.font.init()
    mk = lambda s, b=False: pygame.font.SysFont("menlo,monospace", s, bold=b)
    return {"s": mk(11), "m": mk(13), "l": mk(16, True), "xl": mk(22, True)}


def _scaled(surf: pygame.Surface, w: int, h: int) -> pygame.Surface:
    key = (id(surf), w, h)
    if key not in _scale_cache:
        _scale_cache[key] = pygame.transform.scale(surf, (w, h))
    return _scale_cache[key]


def text(surf, font, s, pos, color=INK, align="left"):
    img = font.render(s, True, color)
    r = img.get_rect()
    if align == "right":
        r.topright = pos
    elif align == "center":
        r.midtop = pos
    else:
        r.topleft = pos
    surf.blit(img, r)
    return r


# ---- the game ----------------------------------------------------------------------------------
def draw_game(surf, sess: Session, x0: int, y0: int, scale: int, show_boxes: bool, label: str, f) -> pygame.Rect:
    """Draw the engine's current state into a 600x150 game window at (x0, y0), magnified `scale` times."""
    e = sess.engine
    S = scale
    rect = pygame.Rect(x0, y0, WIDTH * S, 150 * S)
    pygame.draw.rect(surf, GAME_BG, rect)
    clip = surf.get_clip()
    surf.set_clip(rect)
    ground = y0 + (GROUND_Y + 47 + 1) * S                               # ground line (dino feet are at y = 140)
    # clouds drift slower than the ground
    for cx, cy in CLOUDS:
        xx = (cx - e.distance * 0.15) % (WIDTH + 100) - 50
        for dx, dy, w in ((0, 6, 44), (10, 0, 24), (24, 3, 28)):
            pygame.draw.rect(surf, (225, 232, 238), ((x0 + (xx + dx) * S), y0 + (cy + dy) * S * 0.9, w * S, 10 * S), border_radius=5)
    pygame.draw.line(surf, (60, 60, 60), (x0, ground), (x0 + WIDTH * S, ground), 2)
    for px, dy, w in PEBBLES:
        xx = (px - e.distance) % WIDTH
        pygame.draw.line(surf, (175, 175, 175), (x0 + xx * S, ground + dy * S * 0.5), (x0 + (xx + w) * S, ground + dy * S * 0.5), 2)

    # obstacles (cacti are drawn tile by tile for groups)
    for ob in e.obstacles:
        ox, oy = x0 + ob.x * S, y0 + ob.y * S
        if ob.kind == PTERODACTYL:
            img = _scaled(sprites.ptero((e.frames // 24) % 2 == 0), 46 * S, 40 * S)
            surf.blit(img, (ox, oy))
        else:
            unit = ob.width // ob.size
            base = sprites.cactus_small() if ob.kind == 0 else sprites.cactus_large()
            img = _scaled(base, unit * S, ob.height * S)
            for i in range(ob.size):
                surf.blit(img, (ox + i * unit * S, oy))

    # the dino
    dead = e.crashed
    near = any(0 < ob.x - (T_START_X + 44) < 130 for ob in e.obstacles)
    eyes = "dead" if dead else "scared" if near else ("blink" if e.frames % 480 < 12 else "normal")
    if e.ducking:
        pose, w, h = ("duck_a" if (e.frames // 12) % 2 == 0 else "duck_b"), 60, 24
    elif e.jumping:
        pose, w, h = "jump", 44, 48
    else:
        pose, w, h = ("run_a" if (e.frames // 12) % 2 == 0 else "run_b"), 44, 48
    img = _scaled(sprites.dino(pose, eyes if pose.startswith("run") or pose == "jump" else "normal"), w * S, h * S)
    surf.blit(img, (x0 + T_START_X * S, y0 + (GROUND_Y + 47) * S - e.height * S - h * S))

    if show_boxes:
        tb = (e.x + 1, e.y + 1)
        pygame.draw.rect(surf, (90, 90, 255), (x0 + tb[0] * S, y0 + tb[1] * S, 42 * S, 45 * S), 1)
        for (bx, by, bw, bh) in (TREX_BOXES_DUCKING if e.ducking else TREX_BOXES_RUNNING):
            pygame.draw.rect(surf, (20, 160, 80), (x0 + (bx + tb[0]) * S, y0 + (by + tb[1]) * S, bw * S, bh * S), 2)
        for ob in e.obstacles[:1]:
            ox, oy = ob.x + 1, ob.y + 1
            pygame.draw.rect(surf, (90, 90, 255), (x0 + ox * S, y0 + oy * S, (ob.width - 2) * S, (ob.height - 2) * S), 1)
            for (cx, cy, cw, ch) in ob.boxes:
                pygame.draw.rect(surf, (225, 50, 50), (x0 + (cx + ox) * S, y0 + (cy + oy) * S, cw * S, ch * S), 2)
        text(surf, f["s"], "boxes: green = dino, red = obstacle (only the FIRST obstacle is collision-tested)", (x0 + 8, y0 + 24), (90, 90, 90))

    # overlays
    text(surf, f["m"], f"score {e.score:05d}   speed {e.speed:.1f}", (x0 + 8, y0 + 6), (60, 60, 60))
    text(surf, f["s"], label, (x0 + WIDTH * S - 8, y0 + 6), (90, 90, 90), "right")
    if e.jumping and e.status == "DUCKING":
        text(surf, f["l"], "STUCK LANDING: ground level but still 'jumping'", (x0 + WIDTH * S // 2, y0 + 30 * S), RED, "center")
    if sess.done:
        shade = pygame.Surface(rect.size, pygame.SRCALPHA)
        shade.fill((255, 255, 255, 150))
        surf.blit(shade, rect.topleft)
        text(surf, f["xl"], sess.crash_text, (rect.centerx, rect.centery - 14), RED, "center")
    surf.set_clip(clip)
    pygame.draw.rect(surf, LINE, rect, 1)
    return rect


# ---- the agent's "mind" ----------------------------------------------------------------------------
def panel(surf, rect, title, f):
    pygame.draw.rect(surf, PANEL, rect, border_radius=8)
    text(surf, f["m"], title, (rect.x + 10, rect.y + 7), MUTE)


def draw_inputs(surf, rect, sess: Session, f):
    panel(surf, rect, "WHAT THE AGENT SEES  (22 numbers)", f)
    obs = sess.obs
    colw = (rect.w - 20) // 2
    groups = [(0, 9, BLUE), (9, 14, GREEN), (14, 19, ORANGE), (19, 22, PURPLE)]
    colour = {}
    for a, b, c in groups:
        for i in range(a, b):
            colour[i] = c
    for i, name in enumerate(OBS_NAMES):
        col, row = (0, i) if i < 11 else (1, i - 11)
        x = rect.x + 10 + col * colw
        y = rect.y + 30 + row * 19
        name = name.replace("obstacle ", "obs ").replace("duck-anim while jumping", "duck anim in air").replace("is pterodactyl", "is ptero").replace("flying height", "fly height")
        text(surf, f["s"], name, (x, y + 1), MUTE)
        bx, bw = x + 125, colw - 125 - 52
        pygame.draw.rect(surf, LINE, (bx, y + 3, bw, 9))
        mid = bx + bw // 2
        v = float(np.clip(obs[i], -1.5, 1.5)) / 1.5
        w = int(abs(v) * bw / 2)
        pygame.draw.rect(surf, colour[i], (mid if v >= 0 else mid - w, y + 3, max(w, 1), 9))
        pygame.draw.line(surf, MUTE, (mid, y + 1), (mid, y + 14))
        text(surf, f["s"], f"{obs[i]:+.2f}", (bx + bw + 4, y + 1), INK)


def draw_actions(surf, rect, sess: Session, f):
    panel(surf, rect, "WHAT IT DECIDES", f)
    y = rect.y + 32
    if sess.probs is not None:
        for i, name in enumerate(ACTION_NAMES):
            p = float(sess.probs[i])
            chosen = i == sess.last_action
            text(surf, f["m"], name, (rect.x + 10, y), INK if chosen else MUTE)
            pygame.draw.rect(surf, LINE, (rect.x + 100, y + 2, rect.w - 170, 12))
            pygame.draw.rect(surf, GREEN if chosen else BLUE, (rect.x + 100, y + 2, int((rect.w - 170) * p), 12))
            text(surf, f["m"], f"{p:5.1%}", (rect.right - 10, y), INK, "right")
            y += 22
    elif sess.kind == "rule":
        text(surf, f["m"], f"hand-written rule says: {sess.rule_action}", (rect.x + 10, y), INK); y += 22
        text(surf, f["s"], "(no probabilities: it follows fixed if/then rules)", (rect.x + 10, y), MUTE); y += 22
    else:
        text(surf, f["m"], f"{sess.kind} agent", (rect.x + 10, y), INK); y += 22
        text(surf, f["s"], "(no network to look inside)", (rect.x + 10, y), MUTE); y += 22
    y = rect.y + 104
    e = sess.engine
    text(surf, f["m"], f"keys held now: {ACTION_NAMES[sess.held]}", (rect.x + 10, y), ORANGE); y += 20
    flags = [("jumping", e.jumping), ("ducking", e.ducking), ("fast fall", e.speed_drop), (f"state {e.status}", True)]
    x = rect.x + 10
    for name, on in flags:
        r = text(surf, f["s"], name, (x, y), INK if on else LINE)
        x = r.right + 12
    y += 24
    text(surf, f["s"], f"decisions {sess.decisions}   time {sess.decisions / 60:.1f} s   display {sess.cfg.hz:.0f} Hz", (rect.x + 10, y), MUTE)


def draw_events(surf, rect, sess: Session, f):
    panel(surf, rect, "WHAT JUST HAPPENED", f)
    y = rect.y + 30
    for ev in list(sess.events)[-8:]:
        bad = ev.startswith("!!") or ev.startswith("CRASH")
        text(surf, f["s"], ev[:62], (rect.x + 10, y), RED if bad else INK)
        y += 16


def draw_brain(surf, rect, sess: Session, f):
    panel(surf, rect, "INSIDE THE NETWORK  (2 hidden layers)", f)
    for k, (h, name) in enumerate(((sess.h1, "layer 1"), (sess.h2, "layer 2"))):
        x0 = rect.x + 12 + k * (rect.w // 2)
        text(surf, f["s"], name + " (128 units)", (x0, rect.y + 28), MUTE)
        if h is None:
            text(surf, f["s"], "n/a for this agent", (x0, rect.y + 48), LINE)
            continue
        cell = max(4, (rect.w // 2 - 30) // 16)
        for i, v in enumerate(h[:128]):
            cx, cy = x0 + (i % 16) * cell, rect.y + 46 + (i // 16) * cell
            t = float(np.clip(v, -1, 1))
            col = (int(60 + 180 * max(t, 0)), int(60 + 40 * (1 - abs(t))), int(60 + 180 * max(-t, 0)))
            pygame.draw.rect(surf, col, (cx, cy, cell - 1, cell - 1))
    text(surf, f["s"], "red = +1, blue = -1, grey = near 0", (rect.x + 12, rect.bottom - 18), MUTE)


def load_curve(run: str):
    p = Path("runs") / run / "eval.csv"
    rows = []
    if p.exists():
        with open(p) as fh:
            for r in csv.DictReader(fh):
                try:
                    rows.append({k: float(v) for k, v in r.items() if k != "frac_cap"})
                except ValueError:
                    pass
    return rows


def draw_curve(surf, rect, rows, marker_M, title, f):
    panel(surf, rect, title, f)
    pl, pr, pt, pb = rect.x + 44, rect.right - 10, rect.y + 30, rect.bottom - 22
    pygame.draw.line(surf, LINE, (pl, pb), (pr, pb))
    if len(rows) < 1:
        text(surf, f["s"], "no evaluation curve for this run", (pl + 10, pt + 20), MUTE)
        return
    xmax = max(r["frames"] for r in rows) * 1.02
    ymax = max(6500, max(r["p90"] for r in rows)) * 1.05
    X = lambda v: pl + v / xmax * (pr - pl)
    Y = lambda v: pb - v / ymax * (pb - pt)
    for v in (0, 2000, 4000, 6000):
        pygame.draw.line(surf, LINE, (pl, Y(v)), (pr, Y(v)))
        text(surf, f["s"], f"{v}", (pl - 4, Y(v) - 6), MUTE, "right")
    for key, col in (("p10", ORANGE), ("mean", GREEN), ("median", BLUE)):
        pts = [(X(r["frames"]), Y(r[key])) for r in rows]
        if len(pts) > 1:
            pygame.draw.lines(surf, col, False, pts, 2)
    if marker_M is not None:
        mx = X(marker_M * 1e6)
        pygame.draw.line(surf, RED, (mx, pt), (mx, pb), 2)
        text(surf, f["s"], f"you are here ({marker_M}M)", (mx + 4, pt), RED)
    text(surf, f["s"], f"{int(xmax / 1e6)}M decisions", (pr, pb + 4), MUTE, "right")
    text(surf, f["s"], "median", (pl + 4, pb + 4), BLUE); text(surf, f["s"], "mean", (pl + 60, pb + 4), GREEN); text(surf, f["s"], "worst 10%", (pl + 105, pb + 4), ORANGE)
