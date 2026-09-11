# Automatic exploration starts: first experiment

This compares the existing Rainbow DQN learner with two ways of collecting
experience. The baseline always starts at the loaded save state. The archive
arm sometimes starts from an emulator snapshot it discovered during its own
play. No room is designated as a goal, and no manual curriculum states are
provided. The normal `main.py` training command is unaffected.

From the repository root:

```bash
uv run python -m scripts.archive_trial \
  --frames-per-arm 64000 \
  --eval-episodes 4
```

Both arms start from the same random weights. `--checkpoint` still works, but
only with a checkpoint trained since the memory planes and state vector were
added. The network input is now 84x84x6 planes plus a 52-wide vector, so
anything older (84x84x7, no vector) fails with an explanatory error. The
checkpoint this document used to name, `20260817_092049`'s `best.keras`, is one
of those. It was also chosen by the old single-episode-reward rule, which on a
later run selected a 76%-random episode 29. Prefer an `episodeNNNN.keras`, or a
`best.keras` from a run whose log has a `score_avg` column.

A run from scratch may spend much of its budget before reaching a key. Use
`--seed` to repeat with independent randomness. Runtime depends on hardware;
the frame budget counts training interaction, with evaluation frames additional.

## What is controlled

Both arms have identical initial network weights (verified by hash), an empty
replay buffer, and the same rewards. Defaults are gamma 0.995, learning rate
0.00025, 10-step returns, frame skip 16, and a gradient update every four
decisions. RND is off. Training epsilon stays at 0.5 so different rollout
lengths do not accidentally change the exploration schedule. These settings
are shared by the two arms; historical runs with other settings are not the
control group. Existing adapter reward rules are retained in both arms.

Both arms feed the network the same observation as `main.py`: stacked frames,
the adapter's two memory planes, and the state vector (22 adapter scalars plus a
one-hot history of the last three actions). `--no-state-vector` drops the vector
in both arms. Gamma stays at this trial's own 0.995, not the 0.997 in
`modelargs.json`, so the trial remains comparable with its earlier design.

Each full trajectory can last up to 8,000 frames from the original start.
Archive starts receive up to 2,000 additional frames, capped by the remaining
trajectory budget. At each rollout boundary, the archive arm selects an
archived start with probability 0.5, once entries exist. Both arms receive
exactly `--frames-per-arm` newly simulated training frames. Restored prefixes
are not counted as newly simulated interaction or added to replay again.

The archive groups normal-play states by room, 32-pixel position cells,
inventory, heart-container capacity, remaining room enemies, and observed
key-consuming door interactions. It does not hash animation timers or enemy
positions. This is a deliberately coarse Zelda-specific representation, not
a complete representation of persistent world changes. The first snapshot
for a cell is retained; at capacity, the most sampled entry is evicted, with
age breaking ties. Selection favors entries with fewer expansion attempts.
The snapshot capacity defaults to 256; the set of seen cell identifiers can
continue growing. The baseline also records an archive for coverage accounting,
but never uses it to choose a start.

Restoring includes the emulator, frame stack, and per-trajectory adapter
history. That covers the memory planes, which are adapter attributes. The state
vector's action history is read off the archived action path, so a restored
start sees the actions that actually led to it. Lifetime tile counts keep advancing. The learner's n-step buffer is
flushed at every rollout boundary, and snapshot jumps never become transitions.
Paths retain actual action indices and frame counts back to the original start.

## Read the results

Outputs go into a new `runs/experiments/<timestamp>__archive_trial/` directory:

- `config.json` and `source/`: parameters, initial checkpoint hash, and source
  copies, including local reward edits.
- `comparison.json`: training and evaluation summaries for both arms.
- `baseline/initial_evaluation.csv`: common checkpoint before training.
- `<arm>/training.csv`: rollout source, start inventory, collected frames,
  key pickups/uses, returns carrying a key, and room transitions.
- `<arm>/final_evaluation.csv`: evaluation from the original start only.
- `<arm>/final.keras` and `final_evaluation.avi`: trained weights and the first
  evaluation episode. Use `--no-video` to skip videos.
- `<arm>/archive_manifest.json`: retained cell metadata and actual action paths.
  Emulator snapshots are currently in memory only and rebuild in each trial.

Evaluation uses epsilon 0.05 with the same seeds in both arms, never trains,
never restores snapshots, and has its own adapter/novelty state. Low exploration
provides some variation instead of counting identical deterministic replays as
independent tests. Evaluation randomness does not advance training's RNG.

Training events count only the new continuation after a restore. An archived
start holding a key is not counted as a key pickup. `rooms_after_key_use` logs
room transitions after a newly observed unlock; it does not assert which door
was crossed. Read the room sequence or recording to verify a particular route.

More discovered cells or key uses during archive rollouts establishes improved
exploration, not continuous playing ability. Look for improvement in the final
entrance-start evaluations as well. Four evaluation episodes and one training
seed provide an initial signal, not a reliable success-rate estimate.

## Checks

```bash
uv run python -m unittest tests.test_zelda_adapter tests.test_exploration_archive -v
uv run ruff check .
uv run mypy
```

The emulator integration checks require the local ROM. They verify that a
restored trajectory reproduces the same frames/RAM, that reward history survives
restoration, that snapshot jumps add no replay transitions, and that evaluation
does not train or modify the training adapter's counts.
