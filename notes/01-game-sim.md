# Milestone 1: game sim + renderer

## Ideas worth remembering

- **Fixed timestep.** `step()` always advances exactly one tick. The game never
  looks at a real clock, so speed is decided by whoever calls `step()`:
  60/sec for a human, thousands/sec for training.
- **Logic and drawing are separate.** `game.py` has no graphics. `render.py` only
  reads the game state. That is why an AI can train with no window and we can
  still watch it later using the same renderer.
- **Determinism.** `TRexGame(seed)` uses its own `random.Random(seed)`, so the same
  seed plus the same actions gives the same game every time.
- **Fairness.** Gaps between obstacles are at least 40 ticks of travel. A jump lasts
  about 33 ticks, so every obstacle can be cleared (milestone 2 tests this).
- **Shrunken hitboxes.** Boxes are 4px smaller than the drawn shapes, so close
  calls feel fair.

## Try it

    python scripts/play.py      # SPACE/UP jump, DOWN duck, R restart, F fullscreen, ESC quit

## Sprites and fullscreen (added after first playtest)

- **Sprites live in `trex/sprites.py`**, drawn from rectangles of palette letters and
  scaled 2x. No image files, so tweaking a sprite means changing a number.
- **Animation is driven by `game.ticks`, not the wall clock.** Running legs, wing
  flaps and blinking will look identical in a replay of the same episode.
- **Personality is a function of game state.** The dino looks scared when an
  obstacle is within 130px, blinks every few seconds, and gets X eyes on a crash.
- **Fullscreen** uses Pygame's `SCALED` mode: the game is always drawn at 600x200
  and stretched to the window. Resolution changes never touch the physics.
