# Zelda Bot

## What is this?

This repo aims to be a working interface and neural network agent to play The Legend of Zelda (1987) on NES.

## What are the goals?

The main goal is, of course, to see if an agent can beat The Legend of Zelda. Given Zelda's non-linear nature, I was interested in seeing how some models typically used to play games with more straightforward objectives would handle this game.

## Isn't Zelda too complicated for an RL agent? What is the plan here?

The current training target is a single isolated combat room (the `monsters` state), where the agent learns to reliably kill enemies. From there the plan is to work up to clearing whole dungeons, and eventually to navigating the overworld. The full plan lives in [games/Zelda/ROADMAP.md](games/Zelda/ROADMAP.md).

## Does this include the game itself?

**NO!** You have to provide the game in .nes format yourself. I will not provide the .nes file in any way, shape, or form.

To integrate the rom into gym-retro, place the correct rom at `games/Zelda/rom.nes`. Running main.py will automatically integrate the game into gym-retro.

SHA1 hash for the Zelda rom used:

    799459548f9636dd263200da494f04058f5540e2

`.gitignore` excludes `games/*/*.nes` so a rom can never be committed by accident. Please leave those patterns in place.

To check that your rom is being picked up before you start a long training run:

    python scripts/integrate.py

## How is the repo laid out?

The training loop is game-agnostic. Everything Zelda-specific — the action set, the reward shaping, when an episode ends, which RAM addresses matter — lives behind a game adapter, so adding another retro game means adding a directory, not editing `main.py`.

    main.py              Training loop (no game-specific logic)
    games/
      base.py            GameAdapter interface that main.py drives
      __init__.py        Adapter registry (load_adapter)
      Zelda/
        adapter.py       Action set, reward shaping, termination, metrics
        data.json        RAM address mappings
        RAM_MAP.md       Verified RAM map + how to find new addresses
        ROADMAP.md       Where this project is going
        ram_search.py    RAM discovery tool
        *.state          Save states (gamestart, level1-8, monsters)
    models/              DQN, DoubleDQN, RainbowDQN
    scripts/             Utility and smoke-test scripts

## What models are provided?

**DQN** — Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). _Playing Atari with Deep Reinforcement Learning._ [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)

**DoubleDQN** — van Hasselt, H., Guez, A., & Silver, D. (2016). _Deep Reinforcement Learning with Double Q-learning._ [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)

**RainbowDQN** — a partial Rainbow: dueling head, prioritized replay, 3-step returns, soft target updates, Huber loss, and Double DQN targets. Noisy Nets and distributional returns are not implemented yet. Hessel, M., et al. (2018). _Rainbow: Combining Improvements in Deep Reinforcement Learning._ [arXiv:1710.02298](https://arxiv.org/abs/1710.02298)

See [models/CLAUDE.md](models/CLAUDE.md) for the network architecture and the interface a new model has to implement.

## How do I run this?

### GPU setup

These models were trained and run using a commercial Nvidia GPU. Using Tensorflow with a GPU on Windows natively is not currently supported, but there is support for Windows Subsystem for Linux (WSL) along with the appropriate driver setup.

* [Instructions for setting up WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
* [Instructions for setting up Tensorflow's GPU support with WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

You can run these using CPU only, but it's going to be **slow**.

### Required libraries

You can install the required libraries using pip and the requirements.txt file:

    pip install -r ./requirements.txt

To confirm the GPU is visible to Tensorflow:

    python scripts/gputest.py

### Linting

Ruff and Mypy are configured in `pyproject.toml`. They aren't needed to train, so they live in a separate requirements file:

    pip install -r requirements-dev.txt
    ruff check .          # add --fix to apply the safe fixes
    mypy

Mypy runs in a deliberately permissive mode: this is numeric code with no annotations, so the aim is to catch real mistakes (typos, bad attribute access, unreachable branches) rather than to enforce full type coverage. `NPY002` is switched off in Ruff on purpose — migrating `np.random.*` to `Generator` would change the RNG stream and therefore every training trajectory, which is a behavioural change rather than a lint fix.

### Training the models

Once gym-retro is set up and the game is integrated, you can train the models by running main.py:

    python main.py --config modelargs.json

The defaults in `modelargs.json` run 500 episodes of RainbowDQN on the `monsters` combat room. That takes roughly 4-8 hours on a 3060 Ti, and kills-per-episode should start climbing somewhere around episode 100-200.

Configuration settings can be set either in modelargs.json or on the command line by specifying --arg and following it with an appropriate value. Command line arguments override the config file. The arguments are listed below:

* **config**: path to the JSON config file to read defaults from. Defaults to `modelargs.json`.
* **game**: the game to train on, matching a directory under `games/`. Defaults to **Zelda**.
* **state**: determines which save state to load the game in. Under the game's folder there are multiple states. **gamestart** starts the player at the beginning of the game, each **level state** (levelx.state) starts at dungeon x with the minimum needed to get to or beat that dungeon, and **monsters** is an isolated combat room. Use only the name of the state, without the file extension. If omitted, the game adapter's default state is used.
* **model**: which agent to train — **DQN**, **DoubleDQN**, or **RainbowDQN**.
* **num_episodes**: the number of episodes for the model to run.
* **learning_rate**: controls how quickly the model updates its parameters during training.
* **discount_factor**: determines the present value of future rewards, balancing immediate rewards against long-term gains.
* **epsilon**: the probability that the agent will take a random action instead of following its current learned policy, encouraging exploration of the environment.
* **epsilon_decay**: the factor by which epsilon is reduced after each episode. This gradual reduction allows the agent to shift from exploration towards exploitation as it learns.
* **epsilon_min**: the minimum value to which epsilon can decay. Ensures that the agent always retains a small probability of taking random actions, which can help prevent the policy from becoming completely deterministic.
* **max_frames**: the maximum amount of frames for the agent to step through in one episode.
* **load_model**: `latest`, or a path to a specific checkpoint. See below.
* **record_freq**: record a video of every Nth episode. Defaults to 25.

Note that the model will be saved under the directory ./saved_models, and per-episode stats are appended to training_log.csv.

### Loading models

To start fresh training without loading any model, just leave `--load_model` off.

To resume from the most recent saved model:

    python main.py --config modelargs.json --load_model latest

To load a specific model file:

    python main.py --config modelargs.json --load_model saved_models/RainbowDQNAgent_episode100_20260304_114032.keras

Checkpoints are saved as `.keras`; older `.h5` files still load. Resuming drops epsilon to `epsilon_min` so the agent exploits what it already learned instead of re-exploring from scratch — pass `--epsilon` explicitly if you want to override that.

## How do I add another game?

1. Create `games/<GameName>/` containing the gym-retro integration files (`data.json`, `scenario.json`, `metadata.json`, `rom.sha`, and your save states).
2. Add `games/<GameName>/adapter.py` with a `get_adapter(state)` factory returning a subclass of `GameAdapter` (see [games/base.py](games/base.py)). The adapter owns the action set, the reward shaping, the termination rules, and any extra columns you want in the training log.
3. Run `python main.py --game <GameName>`.

`main.py` does not need to change. `games/Zelda/adapter.py` is the reference implementation.

## To-do list

* Finish Rainbow: Noisy Nets to replace fixed epsilon-greedy, and distributional returns (C51 or QR-DQN).
* Tune the reward function to encourage exploration once training moves past a single room.
* Add intrinsic curiosity (RND) for dungeon navigation.
* Create a function to allow the user to update values (such as giving the agent specific weapons, number of rupees, etc.) defined in a JSON file.
* Create a more robust model load feature and allow user to load model based on arguments.

## Sources

*Some RAM addresses were sourced from:*
* [DataCrystal](https://datacrystal.romhacking.net/wiki/The_Legend_of_Zelda:RAM_map)
* [Zophar's Domain](https://zeldit.zophar.net/hack.html)
* [Trax's Rom Hacking](https://www.bwass.org/romhack/zelda1/zelda1rammap.txt)

Addresses that were verified (or corrected) against an actual recording are documented in [games/Zelda/RAM_MAP.md](games/Zelda/RAM_MAP.md).
