"""
RAM discovery tool for The Legend of Zelda (NES).

Runs a random agent while recording all 2KB of NES system RAM every frame,
then analyzes the recording offline to find and verify variable addresses.

Subcommands:
  record     Run a random agent and dump RAM + info to an .npz file
  analyze    Classify every address (counters, flags, timers) and annotate
             with the community (DataCrystal) RAM map
  correlate  Find addresses that change exactly when a chosen event fires
             (e.g. an enemy dies) — the "cheat search" step
  verify     Check every variable in games/Zelda/data.json against the recording
             and the community map, flagging suspicious entries

Typical workflow:
  python games/Zelda/ram_search.py record --state monsters --frames 6000
  python games/Zelda/ram_search.py analyze ram_dumps/monsters.npz
  python games/Zelda/ram_search.py correlate ram_dumps/monsters.npz --event dec:0x34E
  python games/Zelda/ram_search.py verify ram_dumps/monsters.npz
"""

import argparse
import json
import os
import sys

import numpy as np

GAME_DIR = os.path.dirname(os.path.abspath(__file__))      # games/Zelda
GAMES_DIR = os.path.dirname(GAME_DIR)                       # games (retro custom path)
REPO_ROOT = os.path.dirname(GAMES_DIR)                      # repo root (for ram_dumps)
DATA_JSON = os.path.join(GAME_DIR, "data.json")

# Community RAM map (DataCrystal, datacrystal.tcrf.net/wiki/The_Legend_of_Zelda/RAM_map).
# Used to annotate analysis output so newly found addresses can be cross-checked.
KNOWN_MAP = {
    0x0D: "full hearts damage taken",
    0x0E: "partial heart damage taken",
    0x0F: "Link move direction (1=E 2=W 4=S 8=N FF=none)",
    0x10: "current level (0=overworld)",
    0x12: "game mode (5=normal 6/7=scrolling 4=finish scroll)",
    0x13: "routine index",
    0x15: "frame counter",
    0x28: "X update trigger",
    0x50: "kills without damage (resets at 10 / on hit)",
    0x70: "Link X",
    0x84: "Link Y",
    0x98: "Link direction (1=E 2=W 4=S 8=N)",
    0xAC: "Link animation",
    0xB9: "Link sword animation",
    0xBA: "sword projectile / magic state",
    0xE0: "game paused (1=yes)",
    0xE8: "screen scrolling direction (0=no 1=E 2=W 4=S 8=N)",
    0xEB: "map location / room (x + 0x10*y)",
    0xEC: "next room during scroll",
    0xF8: "P1 buttons held",
    0xFA: "P1 buttons pressed",
    0xFC: "subscreen Y-scroll (pause menu)",
    0xFD: "subscreen X-scroll (pause menu)",
    0x034D: "room clear counter (dungeons)",
    0x034E: "(undocumented; observed: enemies alive in room)",
    0x0394: "Link sub-tile",
    0x03A8: "Link subpixel",
    0x049E: "Link colliding tile (0x26=empty)",
    0x0513: "candle used this screen",
    0x0526: "cave return screen",
    0x052A: "enemies-killed counter",
    0x052E: "sword disabled by red bubble",
    0x0606: "sound effects trigger",
    0x0627: "killed enemy count (current screen)",
    0x0656: "selected B item",
    0x0657: "sword type",
    0x0658: "bombs",
    0x0659: "arrow type",
    0x065A: "bow",
    0x065B: "candle",
    0x065C: "whistle",
    0x065D: "food",
    0x065E: "potion",
    0x065F: "magic rod",
    0x0660: "raft",
    0x0661: "magic book",
    0x0662: "ring",
    0x0663: "ladder",
    0x0664: "magical key",
    0x0665: "power bracelet",
    0x0666: "letter",
    0x0667: "compass bitfield",
    0x0668: "map bitfield",
    0x066C: "clock",
    0x066D: "rupees",
    0x066E: "keys",
    0x066F: "hearts (low nibble=filled, high=containers-1)",
    0x0670: "partial heart (0=empty 1-7F=half 80-FF=full)",
    0x0671: "triforce bitfield",
    0x0674: "boomerang",
    0x0675: "magical boomerang",
    0x0676: "magic shield",
    0x067C: "max bombs",
    0x067D: "rupees to add",
    0x067E: "rupees to subtract",
}
# Per-slot object tables (slot 0 = Link, slots 1-6 = enemies, rest = projectiles/items)
for slot in range(1, 7):
    KNOWN_MAP[0x70 + slot] = f"enemy #{slot} X"
    KNOWN_MAP[0x84 + slot] = f"enemy #{slot} Y"
    KNOWN_MAP[0x98 + slot] = f"enemy #{slot} direction"
    KNOWN_MAP[0x29 + slot - 1] = f"enemy #{slot} action countdown"
    KNOWN_MAP[0xAC + slot] = f"enemy #{slot} dropped item type"
for slot in range(0x0350, 0x035C):
    KNOWN_MAP[slot] = f"object type slot {slot - 0x0350}"

# Address ranges that change constantly but carry no game-state signal.
NOISE_RANGES = [
    (0x0000, 0x0008, "sprite scratch"),
    (0x0066, 0x0070, "music engine"),
    (0x05F0, 0x0620, "audio engine"),
    (0x0600, 0x0620, "audio engine"),
    (0x0301, 0x0340, "PPU update buffer"),
]


def annotate(addr):
    if addr in KNOWN_MAP:
        return KNOWN_MAP[addr]
    for lo, hi, name in NOISE_RANGES:
        if lo <= addr < hi:
            return f"({name})"
    return ""


def is_noise(addr):
    return any(lo <= addr < hi for lo, hi, _ in NOISE_RANGES)


def make_env(state):
    import retro
    retro.data.Integrations.add_custom_path(GAMES_DIR)
    return retro.make("Zelda", state=state, inttype=retro.data.Integrations.ALL)


# Button order: ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
# Random policy explores movement + sword (A) + item (B); never START/SELECT.
RANDOM_COMBOS = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # up
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # down
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # left
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # right
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # A (sword)
    [0, 0, 0, 0, 1, 0, 0, 0, 1],  # up + A
    [0, 0, 0, 0, 0, 1, 0, 0, 1],  # down + A
    [0, 0, 0, 0, 0, 0, 1, 0, 1],  # left + A
    [0, 0, 0, 0, 0, 0, 0, 1, 1],  # right + A
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # B (item)
]


def cmd_record(args):
    env = make_env(args.state)
    env.reset()

    ram_size = args.ram_size
    rams = np.zeros((args.frames, ram_size), dtype=np.uint8)
    rng = np.random.default_rng(args.seed)

    action = np.array(RANDOM_COMBOS[rng.integers(len(RANDOM_COMBOS))], dtype=np.uint8)
    hold = 0
    resets = 0
    for f in range(args.frames):
        if hold <= 0:
            action = np.array(RANDOM_COMBOS[rng.integers(len(RANDOM_COMBOS))], dtype=np.uint8)
            hold = rng.integers(4, 16)  # hold a combo 4-15 frames, like real play
        hold -= 1

        _, _, terminated, truncated, _ = env.step(action)
        rams[f] = env.get_ram()[:ram_size]

        if terminated or truncated:
            env.reset()
            resets += 1
        if f % 1000 == 0:
            print(f"frame {f}/{args.frames} (resets: {resets})")

    out = args.out or os.path.join(REPO_ROOT, "ram_dumps", f"{args.state}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, rams=rams, state=args.state, seed=args.seed)
    env.close()
    print(f"Saved {args.frames} frames x {ram_size} bytes to {out} ({resets} env resets)")


def load_rams(path):
    data = np.load(path, allow_pickle=True)
    return data["rams"]


def classify(series):
    """Classify one address's time series into a coarse behavioral category."""
    diffs = np.diff(series.astype(np.int16))
    changes = np.count_nonzero(diffs)
    if changes == 0:
        return "constant", changes
    nonzero = diffs[diffs != 0]
    uniq = len(np.unique(series))
    change_rate = changes / len(diffs)
    if change_rate > 0.5:
        return "timer/animation", changes
    ups = np.count_nonzero(nonzero > 0)
    downs = np.count_nonzero(nonzero < 0)
    # counters mostly increment, with occasional wrap/reset drops
    if ups > 0 and downs <= max(1, ups // 5) and uniq > 1:
        return "counter(up)", changes
    if downs > 0 and ups <= max(1, downs // 5):
        return "counter(down)", changes
    if uniq == 2:
        return "flag", changes
    return "variable", changes


def cmd_analyze(args):
    rams = load_rams(args.dump)
    print(f"Loaded {rams.shape[0]} frames x {rams.shape[1]} addresses\n")
    rows = []
    for addr in range(rams.shape[1]):
        kind, changes = classify(rams[:, addr])
        if kind == "constant":
            continue
        if args.skip_noise and is_noise(addr):
            continue
        series = rams[:, addr]
        rows.append((addr, kind, changes, len(np.unique(series)),
                     int(series.min()), int(series.max()), annotate(addr)))

    rows.sort(key=lambda r: r[2])
    print(f"{'addr':>6} {'hex':>6} {'kind':16} {'chgs':>6} {'uniq':>5} {'min':>4} {'max':>4}  known-as")
    for addr, kind, changes, uniq, lo, hi, note in rows:
        if args.kind and args.kind not in kind:
            continue
        print(f"{addr:>6} {addr:>#6x} {kind:16} {changes:>6} {uniq:>5} {lo:>4} {hi:>4}  {note}")
    print(f"\n{len(rows)} non-constant addresses "
          f"({rams.shape[1] - len(rows)} constant or filtered)")


def parse_event(spec, rams):
    """Event spec: dec:0xADDR | inc:0xADDR | chg:0xADDR — frames where that happens."""
    op, addr = spec.split(":")
    addr = int(addr, 0)
    diffs = np.diff(rams[:, addr].astype(np.int16))
    if op == "dec":
        mask = diffs < 0
    elif op == "inc":
        mask = diffs > 0
    elif op == "chg":
        mask = diffs != 0
    else:
        raise ValueError(f"unknown event op {op!r} (use dec/inc/chg)")
    return np.flatnonzero(mask), addr


def cmd_correlate(args):
    rams = load_rams(args.dump)
    events, ev_addr = parse_event(args.event, rams)
    print(f"Event {args.event} ({annotate(ev_addr) or 'unannotated'}): "
          f"{len(events)} occurrences in {rams.shape[0]} frames\n")
    if len(events) == 0:
        print("No occurrences — record a longer session or pick another event.")
        return

    n = rams.shape[0] - 1
    event_mask = np.zeros(n, dtype=bool)
    # allow +/- window frames of slack between the event and the side effect
    for e in events:
        event_mask[max(0, e - args.window):e + args.window + 1] = True

    rows = []
    for addr in range(rams.shape[1]):
        if addr == ev_addr or (args.skip_noise and is_noise(addr)):
            continue
        diffs = np.diff(rams[:, addr].astype(np.int16)) != 0
        total = diffs.sum()
        if total == 0:
            continue
        hits = np.count_nonzero(diffs & event_mask)
        covered = sum(1 for e in events
                      if diffs[max(0, e - args.window):e + args.window + 1].any())
        recall = covered / len(events)       # fraction of events this addr reacted to
        precision = hits / total             # how exclusive to the event it is
        if recall >= args.min_recall and precision >= args.min_precision:
            f1 = 2 * precision * recall / (precision + recall)
            rows.append((f1, precision, recall, addr, total))

    rows.sort(reverse=True)
    print(f"{'addr':>6} {'hex':>6} {'prec':>6} {'recall':>6} {'chgs':>6}  known-as")
    for f1, prec, rec, addr, total in rows[:args.top]:
        print(f"{addr:>6} {addr:>#6x} {prec:>6.2f} {rec:>6.2f} {total:>6}  {annotate(addr)}")
    if not rows:
        print("Nothing matched — loosen --min-recall/--min-precision or widen --window.")


def cmd_verify(args):
    rams = load_rams(args.dump)
    with open(DATA_JSON) as f:
        info_vars = json.load(f)["info"]

    print(f"Verifying {len(info_vars)} data.json variables against "
          f"{rams.shape[0]} recorded frames and the community RAM map\n")
    print(f"{'variable':30} {'addr':>6} {'hex':>6} {'uniq':>5} {'min':>4} {'max':>4}  community map says")
    for name, spec in sorted(info_vars.items()):
        addr = spec["address"]
        if addr >= rams.shape[1]:
            print(f"{name:30} {addr:>6} {addr:>#6x}  -- outside recorded range --")
            continue
        series = rams[:, addr]
        note = annotate(addr) or "?? not in community map"
        flag = ""
        # decimal/hex confusion heuristic: the same digits read as hex IS mapped
        try:
            as_hex = int(str(addr), 16)
            if addr not in KNOWN_MAP and as_hex != addr and as_hex in KNOWN_MAP:
                flag = f"  <-- SUSPECT: {addr} looks like ${addr} hex = {as_hex} ({KNOWN_MAP[as_hex]})"
        except ValueError:
            pass
        print(f"{name:30} {addr:>6} {addr:>#6x} {len(np.unique(series)):>5} "
              f"{series.min():>4} {series.max():>4}  {note}{flag}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="record RAM while a random agent plays")
    pr.add_argument("--state", default="monsters")
    pr.add_argument("--frames", type=int, default=6000)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--ram-size", type=int, default=2048,
                    help="bytes of RAM to record per frame (2048 = full NES system RAM)")
    pr.add_argument("--out", default=None)
    pr.set_defaults(fn=cmd_record)

    pa = sub.add_parser("analyze", help="classify every changing address")
    pa.add_argument("dump")
    pa.add_argument("--kind", default=None, help="filter by category substring, e.g. counter")
    pa.add_argument("--skip-noise", action="store_true", default=True)
    pa.add_argument("--no-skip-noise", dest="skip_noise", action="store_false")
    pa.set_defaults(fn=cmd_analyze)

    pc = sub.add_parser("correlate", help="find addresses that change when an event fires")
    pc.add_argument("dump")
    pc.add_argument("--event", required=True, help="dec:0xADDR | inc:0xADDR | chg:0xADDR")
    pc.add_argument("--window", type=int, default=2, help="frames of slack around the event")
    pc.add_argument("--min-recall", type=float, default=0.5)
    pc.add_argument("--min-precision", type=float, default=0.05)
    pc.add_argument("--top", type=int, default=40)
    pc.add_argument("--skip-noise", action="store_true", default=True)
    pc.add_argument("--no-skip-noise", dest="skip_noise", action="store_false")
    pc.set_defaults(fn=cmd_correlate)

    pv = sub.add_parser("verify", help="check data.json variables against a recording")
    pv.add_argument("dump")
    pv.set_defaults(fn=cmd_verify)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
