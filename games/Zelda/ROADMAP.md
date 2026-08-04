# Zelda-Bot Roadmap

Current training target: single isolated combat room (`monsters` state).
The agent learns to kill enemies in one room without navigating the dungeon.

---

## Phase 1 — Single-Room Combat (current)

**Goal:** reliable enemy elimination in the `monsters` state.

**Implemented in `RainbowDQN`:**
- Dueling network head (V + A decomposition)
- Prioritized experience replay (alpha=0.6, beta annealing)
- 3-step returns (faster reward propagation)
- Soft target updates via Polyak averaging (tau=0.0005)
- Huber loss (stable training through large early TD errors)
- Double DQN targets (reduces Q-value overestimation)

**Remaining single-room work:**
- Noisy Nets (learnable ε-per-parameter noise) to replace fixed ε-greedy
- Distributional returns (C51 or QR-DQN) for full Rainbow

---

## Phase 2 — Dungeon Navigation

**Goal:** clear a full dungeon (level 1–8) from a dungeon entry state.

### Key changes

| Area | Change |
|---|---|
| Reward | Flip the `-5.0` new-room penalty; replace with a small positive signal for entering unseen rooms |
| Intrinsic curiosity | Add Random Network Distillation (RND): reward = prediction error of a fixed random network, incentivises exploration of novel states |
| State representation | Augment pixel input with RAM variables (room ID, enemy count, Link health) as a separate dense branch fed into the dueling head |
| Curriculum | Start from `level1`; gate advancement to deeper levels on a rolling-window success threshold (e.g., mean episode reward > X over last 20 episodes) |
| Episode length | Increase `max_frames` to 5000–10000 to allow multi-room traversal |

### Architecture sketch

```
pixel branch: Conv → Conv → Conv → Flatten → Dense(512)
ram branch:   Dense(64) → Dense(64)
combined:     Concatenate → Dueling head
```

### RND module (separate from the main agent)

```python
class RNDModule:
    # target_net: fixed random Conv network (never trained)
    # predictor_net: trained to predict target_net output
    # intrinsic_reward = MSE(predictor(s), target(s))
```

Intrinsic reward is added to extrinsic reward with a scaling coefficient
(start at 0.1, tune based on exploration coverage).

---

## Phase 3 — Full Overworld + Dungeon Integration

**Goal:** navigate from the overworld to any dungeon and clear it end-to-end.

### Hierarchical RL

Two-level architecture:
- **Manager (high-level):** selects subgoals (e.g., "reach dungeon 3 entrance", "collect key")
- **Worker (low-level):** executes actions to achieve the current subgoal

Manager is trained with a slower update rate and a longer time horizon.
Worker receives the subgoal as additional input and is rewarded for reaching it.

Candidate framework: HIRO (Data-Efficient Hierarchical Reinforcement Learning).

### Curriculum learning pipeline

1. Master overworld navigation (reach any dungeon entrance)
2. Master individual dungeons (already done by Phase 2)
3. Compose: navigate overworld → enter dungeon → clear dungeon

Each stage uses a replay buffer seeded with demonstrations from the previous stage
(replay-buffer pre-filling from saved checkpoints).

### Multi-task reward shaping

| Task | Reward components |
|---|---|
| Overworld | +map coverage, +dungeon reached, -time |
| Dungeon | +enemy kill, +item pickup, +boss kill, -death, -time |
| Combined | weighted sum; weight schedule annealed across training |

---

## Long-Term — World Models

**Goal:** sample-efficient planning without real-time emulator interaction.

### DreamerV3-style approach

1. **World model:** learns a compact latent representation of game state and a dynamics model that predicts next latent state + reward from current latent + action.
2. **Actor-critic:** trained entirely in imagination (rollouts inside the world model), greatly reducing real environment interactions.
3. **Benefits for Zelda:** can plan across room boundaries in latent space without the emulator running; supports very long horizons.

### Prerequisites

- Stable single-room and dungeon agents (Phases 1–2) to provide diverse replay data
- Variational autoencoder or RSSM (Recurrent State Space Model) for the world model
- Separate actor and critic networks operating in latent space

### Implementation notes

- Use `stable-retro` only for data collection; world model training can happen offline on stored transitions
- Latent space dimensionality: 32–64 categorical variables (DreamerV3 default)
- Training schedule: alternate between world model updates and imagination-based policy updates

---

## Open Questions

- Does the `-5.0` new-room penalty need to stay for Phase 2, or can it be replaced entirely once the agent has learned room-level combat?
- Should RAM state (43 variables in `data.json`) be included as auxiliary input from Phase 1, or only added in Phase 2?
- Noisy Nets vs. ε-greedy: worth switching before Phase 2, or address in Phase 2 refactor?
