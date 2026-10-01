/*
 * Rule-based agent for the REAL Chrome dino game (chrome://dino, chromedino.com).
 * A port of trex/agents/rule_based.py - keep the two in sync.
 *
 * HOW TO RUN
 *   1. Open the game page, open DevTools (Cmd+Option+J), go to the Console tab.
 *   2. Paste this whole file and press Enter.
 *   3. Run:   dinoAgent.start({ episodes: 3, maxScore: 1500 })
 *      Keep the tab visible and in front: browsers throttle hidden tabs.
 *   4. Stop any time with dinoAgent.stop().  Results: dinoAgent.results
 *
 * It presses keys the same way a person would (dispatches keydown/keyup events);
 * it only READS the game's state to decide.
 *
 * TWO LESSONS BUILT IN (see notes/04-rule-based-agent.md)
 *   - Wide obstacles (groups of 2-3 cacti) need a LATER jump (widthAware).
 *   - The game's reported speed is not the real speed. On a 120 Hz screen obstacles move
 *     ~14% slower than `currentSpeed` says (whole-pixel steps per frame), so we MEASURE
 *     how fast obstacles actually move instead of trusting the number.
 */
(() => {
  const DEFAULTS = { lead: 12, widthAware: true, arcCenter: 16.5 };
  const DINO_W = 44, HITBOX_SHRINK = 4;
  const KEY = { JUMP: 32, DUCK: 40, RESTART: 13 };

  // ---- The decision rule (pure function; same logic as rule_based.py) ---------------
  // obs: { hasObstacle, speed (px per 1/60 s), dist (px from dino's nose), width (px),
  //        flyY (px off the ground), onGround }
  function decide(obs, opts = {}) {
    const { lead, widthAware, arcCenter } = { ...DEFAULTS, ...opts };
    if (!obs.hasObstacle) return 'NOOP';
    const arrivesIn = obs.dist / obs.speed;                 // distance -> time
    if (obs.flyY >= 45) return 'NOOP';                      // high pterodactyl: flies over us
    if (obs.flyY >= 20) return arrivesIn < lead ? 'DUCK' : 'NOOP';   // mid: duck under
    // cactus or low pterodactyl: jump. A wide obstacle needs the high part of the arc to
    // cover it for longer, so jump later: aim the arc's center at the overlap's center.
    const l = widthAware
      ? arcCenter - (obs.width + DINO_W - 2 * HITBOX_SHRINK) / (2 * obs.speed)
      : lead;
    if (obs.onGround && arrivesIn >= 0 && arrivesIn < l) return 'JUMP';
    return 'NOOP';
  }

  // ---- Measure the real world speed ---------------------------------------------------
  // Track every obstacle's x between frames; the average px moved per 1/60 s is the true speed.
  const motion = { v: null, seen: new WeakMap() };
  function updateMotion(r, now) {
    for (const o of r.horizon.obstacles) {
      const p = motion.seen.get(o);
      if (p && now - p.t > 1) {
        const v = ((p.x - o.xPos) / (now - p.t)) * (1000 / 60);
        motion.v = motion.v === null ? v : motion.v + 0.05 * (v - motion.v);   // smooth it
      }
      motion.seen.set(o, { x: o.xPos, t: now });
    }
  }
  function effectiveSpeed(r) {
    const nominal = r.currentSpeed;
    if (motion.v === null) return nominal * 0.87;           // before any measurement
    return Math.min(nominal * 1.05, Math.max(nominal * 0.5, motion.v));
  }

  // ---- Read the game's state and translate it into the same terms as our sim --------
  function observe(r) {
    const t = r.tRex;
    const noseX = t.xPos + t.config.WIDTH;
    const groundBottom = t.groundYPos + t.config.HEIGHT;   // y of the ground line
    const o = r.horizon.obstacles.find((ob) => ob.xPos + ob.width > t.xPos);
    const base = {
      speed: effectiveSpeed(r), nominalSpeed: r.currentSpeed, onGround: !t.jumping,
      dinoY: groundBottom - t.config.HEIGHT - t.yPos,
    };
    if (!o) return { ...base, hasObstacle: false };
    return {
      ...base,
      hasObstacle: true,
      dist: o.xPos - noseX,
      flyY: groundBottom - (o.yPos + o.typeConfig.height),  // height of its underside
      type: o.typeConfig.type,
      size: o.size,                                         // cacti in the group
      width: o.width,
    };
  }

  // ---- Keyboard ------------------------------------------------------------------
  function key(type, code) {
    const e = new Event(type, { bubbles: true, cancelable: true });
    e.keyCode = code;
    e.which = code;
    document.dispatchEvent(e);
  }

  const S = { running: false, results: [], jumpDown: false, jumpFrame: 0, duckDown: false };

  function apply(action, r, frame) {
    const t = r.tRex;
    // Jump: hold the key until we land (letting go early cuts a jump short).
    if (action === 'JUMP' && !S.jumpDown) {
      key('keydown', KEY.JUMP);
      S.jumpDown = true;
      S.jumpFrame = frame;
    } else if (S.jumpDown && !t.jumping && frame - S.jumpFrame >= 2) {
      key('keyup', KEY.JUMP);
      S.jumpDown = false;
    }
    // Duck: on the ground it crouches; in the air it makes you fall faster.
    if (action === 'DUCK') {
      if (t.jumping || !t.ducking) key('keydown', KEY.DUCK);
      S.duckDown = true;
    } else if (S.duckDown) {
      key('keyup', KEY.DUCK);
      S.duckDown = false;
    }
  }

  function releaseKeys() {
    if (S.jumpDown) key('keyup', KEY.JUMP);
    if (S.duckDown) key('keyup', KEY.DUCK);
    S.jumpDown = S.duckDown = false;
  }

  // chrome://dino has r.playing. Older copies (chromedino.com) don't. There `activated` is not
  // reset after a restart and `paused` stays true while the game runs, so neither can be
  // trusted. On a freshly loaded page the loop already ticks before the dino has started, so
  // "loop running" is not enough either: also require that the dino has actually run
  // (distanceRan > 0), which is false on the start screen and true a moment after any start.
  const isPlaying = (r) =>
    r.playing !== undefined
      ? r.playing
      : !!(r.isRunning && r.isRunning() && !r.crashed && r.distanceRan > 0);

  const score = (r) => Math.ceil(r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan)));

  // ---- Episode loop --------------------------------------------------------------
  function start(options = {}) {
    const { episodes = 3, maxScore = 1500, ...opts } = options;
    const Runner = window.Runner;
    if (!Runner || !Runner.instance_) throw new Error('No Runner.instance_ here: open the dino game');
    S.running = true;
    S.results = [];
    let frame = 0, lastObs = null, lastAction = 'NOOP', restartAt = 0, lastKick = 0;
    console.log(`[dinoAgent] ${episodes} episodes, stop at score ${maxScore}`, { ...DEFAULTS, ...opts });

    function finish(outcome) {
      const r = Runner.instance_;
      const res = {
        episode: S.results.length + 1, outcome, score: score(r), frames: frame,
        speed: +r.currentSpeed.toFixed(2), measuredSpeed: motion.v && +motion.v.toFixed(2),
        distance: Math.round(r.distanceRan),
      };
      if (outcome === 'crash' && lastObs) {
        Object.assign(res, {
          killer: lastObs.type, group: lastObs.size, killerWidth: lastObs.width,
          distAtLastFrame: Math.round(lastObs.dist), flyY: Math.round(lastObs.flyY),
          action: lastAction, wasOnGround: lastObs.onGround, dinoY: Math.round(lastObs.dinoY),
        });
      }
      S.results.push(res);
      console.log('[dinoAgent]', JSON.stringify(res));
      releaseKeys();
      frame = 0; lastObs = null; lastAction = 'NOOP';
    }

    function tick() {
      if (!S.running) return;
      const r = Runner.instance_;
      const now = performance.now();
      updateMotion(r, now);

      if (r.crashed) {
        if (!restartAt) {                                  // just crashed
          if (frame > 0) finish('crash');                  // ignore crashes we weren't playing
          if (S.results.length >= episodes) return done();
          restartAt = now + 1300;
        } else if (now >= restartAt) {
          key('keydown', KEY.RESTART); key('keyup', KEY.RESTART);   // Enter restarts
          restartAt = 0;
        }
      } else if (!isPlaying(r)) {
        if (now - lastKick > 1000) {                       // press jump to start the game
          key('keydown', KEY.JUMP); key('keyup', KEY.JUMP); lastKick = now;
        }
      } else {
        frame++;
        lastObs = observe(r);
        lastAction = decide(lastObs, opts);
        apply(lastAction, r, frame);
        if (score(r) >= maxScore) {                        // survived long enough
          finish('survived');
          if (S.results.length >= episodes) { r.gameOver(); return done(); }
          r.gameOver(); restartAt = now + 1300;
        }
      }
      requestAnimationFrame(tick);
    }

    function done() {
      S.running = false;
      const crashed = S.results.filter((x) => x.outcome === 'crash').length;
      console.log(`[dinoAgent] done: ${S.results.length - crashed} survived, ${crashed} crashed`);
      console.table(S.results);
    }
    requestAnimationFrame(tick);
  }

  function stop() { S.running = false; releaseKeys(); console.log('[dinoAgent] stopped'); }

  window.dinoAgent = {
    start, stop, decide, observe, motion,
    get results() { return S.results; },
    get running() { return S.running; },
  };
  console.log('[dinoAgent] ready. Run: dinoAgent.start({ episodes: 3, maxScore: 1500 })');
})();
