#!/usr/bin/env python3
"""Mint a new save state from an existing one by writing RAM variables.

Curriculum tool. Some behaviours are unreachable by exploration — on Zelda's
level 1 the agent reached a key in 49 of 90 episodes and spent one in 0 of 314
across every run, so `key_used` and everything behind the locked door were
rewards it had never once received. A value function cannot learn from a reward
it has never observed, and no amount of reward tuning changes that. Starting the
agent from a state where the behaviour *is* reachable does.

    # entrance room, already holding a key: "walk up and open the door" is now
    # ~100 frames from a reward instead of a four-step chain
    uv run scripts/make_state.py --from level1 --set Keys=1 --name level1_key

    # several at once, and a deeper curriculum step
    uv run scripts/make_state.py --from level1 --set Keys=2 --set Bombs=8 \
        --name level1_stocked

Variable names are whatever the game's data.json defines, so `--set` can only
reach values the integration already maps. Writes games/<game>/<name>.state.
"""

import argparse
import gzip
import os
import sys

import numpy as np
import retro

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_DIR = os.path.join(REPO_ROOT, "games")

# Frames to run after writing RAM, so the emulator renders a frame consistent
# with the new values before the state is captured. Kept small: every frame is
# also a frame of game logic that could undo the write, which is why the values
# are re-applied and then verified below.
SETTLE_FRAMES = 4


def parse_set(pairs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--set expects VAR=VALUE, got {p!r}")
        name, _, value = p.partition("=")
        try:
            out[name.strip()] = int(value, 0)
        except ValueError:
            raise SystemExit(f"--set value must be an integer, got {value!r}") from None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="Zelda", help="directory under games/ (default Zelda)")
    ap.add_argument("--from", dest="base", required=True, help="existing state to start from")
    ap.add_argument("--set", action="append", default=[], metavar="VAR=VALUE",
                    help="RAM variable from the game's data.json (repeatable)")
    ap.add_argument("--name", required=True, help="name of the state to write")
    ap.add_argument("--settle", type=int, default=SETTLE_FRAMES,
                    help=f"frames to run after writing (default {SETTLE_FRAMES})")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing state; refuses without this")
    args = ap.parse_args()

    overrides = parse_set(args.set)
    out_path = os.path.join(GAMES_DIR, args.game, f"{args.name}.state")
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(f"{out_path} already exists. Pass --force to overwrite it.")

    retro.data.Integrations.add_custom_path(GAMES_DIR)
    env = retro.make(args.game, state=args.base,
                     inttype=retro.data.Integrations.ALL, render_mode=None)
    noop = np.zeros(env.action_space.shape, dtype=np.uint8)
    env.reset()

    # set_value raises a bare KeyError on an unmapped name, which reads as a
    # crash rather than a usage error — check up front and list what is valid.
    known = set(env.data.lookup_all())
    unknown = [n for n in overrides if n not in known]
    if unknown:
        env.close()
        raise SystemExit(f"not in {args.game}'s data.json: {unknown}\n"
                         f"available: {', '.join(sorted(known))}")

    # Apply, settle, then re-apply: a settle frame is also a frame of game logic
    # and can overwrite what was just written.
    for name, value in overrides.items():
        env.data.set_value(name, value)
    info = {}
    for _ in range(max(args.settle, 1)):
        _, _, _, _, info = env.step(noop)
        for name, value in overrides.items():
            env.data.set_value(name, value)
    _, _, _, _, info = env.step(noop)

    # Verify before writing anything — a silently-ignored variable name would
    # otherwise produce a state that looks fine and behaves like the original.
    bad = {n: (v, int(info[n])) for n, v in overrides.items()
           if n in info and int(info[n]) != v}
    if bad:
        env.close()
        detail = ", ".join(f"{n}: wanted {w}, got {g}" for n, (w, g) in bad.items())
        raise SystemExit(f"values did not stick ({detail}). The game may recompute them.")

    blob = env.em.get_state()
    env.close()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wb") as fh:
        fh.write(blob)

    # Load it back through retro rather than trusting the write.
    env = retro.make(args.game, state=args.name,
                     inttype=retro.data.Integrations.ALL, render_mode=None)
    env.reset()
    _, _, _, _, check = env.step(noop)
    env.close()
    print(f"wrote {out_path} ({len(blob):,} bytes)")
    print(f"  from: {args.base}")
    for name, value in overrides.items():
        got = int(check[name])
        print(f"  {name}: {value}  -> reloads as {got}" + ("" if got == value else "   MISMATCH"))
    if any(int(check[n]) != v for n, v in overrides.items()):
        sys.exit(1)


if __name__ == "__main__":
    main()
