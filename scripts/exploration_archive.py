"""Bounded emulator snapshots for an automatic-start experiment.

This controls data collection, not the DQN update. No room is a designated
target. Entries are created from observed play, and selected by how many times
we have tried exploring from them. The adapter's lifetime familiarity counts
belong to training and must never be rolled back with the emulator.
"""

import copy
import json
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from games.Zelda.adapter import BITFIELD_ITEMS, MAJOR_ITEMS, NORMAL_MODE, ZeldaAdapter

Cell = tuple[Any, ...]
ActionPath = tuple[tuple[int, int], ...]
DoorHistory = frozenset[tuple[int, int, int, int]]
GLOBAL_ADAPTER_FIELDS = {"_tile_counts", "_kill_history"}


def cell_for(info: dict[str, Any], doors: DoorHistory) -> Cell | None:
    """Coarse position, inventory, and observed door/combat progress.

    Ignore animation frames, timers, transient enemy positions, and cumulative
    kill counts. Door history is inferred from observed key consumption, not a
    map of where to go. It distinguishes pre-key and post-unlock zero-key states.
    This is an approximation, not a complete Zelda world-state representation.
    """
    if info["Game Mode"] != NORMAL_MODE:
        return None
    inventory = tuple(int(info[name]) for name in ["Keys", "Bombs", *MAJOR_ITEMS, *BITFIELD_ITEMS])
    alive = max(0, int(info["Enemies Spawned In Room"]) - int(info["Enemies Killed Current Room"]))
    return (int(info["Level"]), int(info["Room"]), int(info["Link X"]) // 32,
            int(info["Link Y"]) // 32, inventory, int(info["Heart Containers"]) >> 4,
            alive, tuple(sorted(doors)))


def update_doors(doors: DoorHistory, old: dict[str, Any], new: dict[str, Any]) -> DoorHistory:
    if int(new["Keys"]) < int(old["Keys"]) and new["Game Mode"] == NORMAL_MODE:
        door = (int(new["Level"]), int(new["Room"]),
                int(new["Link X"]) // 32, int(new["Link Y"]) // 32)
        return doors | {door}
    return doors


@dataclass
class Entry:
    cell: Cell
    emulator: bytes
    raw: np.ndarray
    frames: tuple[np.ndarray, ...]
    adapter_state: dict[str, Any]
    actions: ActionPath
    doors: DoorHistory
    elapsed_frames: int
    selections: int = 0

    def restore(self, env: Any, adapter: ZeldaAdapter) -> tuple[np.ndarray, deque]:
        """Restore without an emulator step or a fabricated replay transition."""
        base = env.unwrapped
        base.em.set_state(zlib.decompress(self.emulator))
        base.data.reset()
        base.data.update_ram()
        base.img = self.raw.copy()
        # Copy mutable per-trajectory state, while retaining the latest global
        # familiarity counts. Repeated restores must not renew novelty rewards.
        for name, value in self.adapter_state.items():
            setattr(adapter, name, copy.deepcopy(value))
        return self.raw.copy(), deque((f.copy() for f in self.frames), maxlen=len(self.frames))


class ExplorationArchive:
    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("archive capacity must be positive")
        self.capacity = capacity
        self.entries: dict[Cell, Entry] = {}
        self.seen: set[Cell] = set()
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def capture(self, env: Any, adapter: ZeldaAdapter, raw: np.ndarray, frames: deque,
                info: dict[str, Any], actions: ActionPath, doors: DoorHistory,
                elapsed_frames: int) -> bool:
        cell = cell_for(info, doors)
        if cell is None or cell in self.seen or adapter.died or adapter.abandoned:
            return False
        self.seen.add(cell)
        if len(self.entries) >= self.capacity:
            # Preserve entries with fewer expansion attempts; insertion order
            # breaks ties. Snapshot memory is bounded even in longer runs.
            victim = max(self.entries, key=lambda c: self.entries[c].selections)
            del self.entries[victim]
        adapter_state = copy.deepcopy({k: v for k, v in vars(adapter).items()
                                       if k not in GLOBAL_ADAPTER_FIELDS})
        self.entries[cell] = Entry(
            cell, zlib.compress(env.unwrapped.em.get_state(), level=1), raw.copy(),
            tuple(f.copy() for f in frames), adapter_state, actions, doors, elapsed_frames,
        )
        return True

    def choose(self) -> Entry:
        if not self.entries:
            raise ValueError("cannot select from an empty archive")
        entries = list(self.entries.values())
        weights = np.array([1.0 / (1 + e.selections) for e in entries])
        entry = entries[int(self.rng.choice(len(entries), p=weights / weights.sum()))]
        entry.selections += 1
        return entry

    def save_manifest(self, path: Path) -> None:
        """Keep actual action ancestry for inspection/reproduction from the root.

        Snapshots are currently in-memory only; a new run rebuilds its archive.
        Each action pair is (action index, emulated frames), including partial
        action windows. Snapshot jumps never appear in these paths.
        """
        path.write_text(json.dumps({
            "unique_cells_seen": len(self.seen),
            "entries": [{"cell": e.cell, "selections": e.selections,
                         "elapsed_frames": e.elapsed_frames, "actions": e.actions}
                        for e in self.entries.values()],
        }, indent=2))
