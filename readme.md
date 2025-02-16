# Zelda Bot

## What is this?

This repo aims to be a working interface and neural network agent to play The Legend of Zelda (1987) on NES.

## What are the goals?

The main goal is, of course, to see if an agent can beat The Legend of Zelda. Given Zelda's non-linear nature, I was interested in seeing how some models typically used to play games with more straightforward objectives would handle this game.

## Isn't Zelda too complicated for an RL agent? What is the plan here?

The initial goal is to have the agents beat all dungeons in the game, with a larger goal of being able to navigate the overworld.

## Does this include the game itself?

**NO!** You have to provide the game in .nes format yourself. I will not provide the .nes file in any way, shape, or form.

To integrate the rom into gym-retro, place the correct rom into the Zelda folder and rename it "rom.nes". Running main.py will automatically integrate the game into gym-retro.

SHA1 hash for the Zelda rom used:

    799459548f9636dd263200da494f04058f5540e2

## What models are provided?

The included DQN model is implemented from the following research paper:

**Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). _Playing Atari with Deep Reinforcement Learning._**  
*arXiv preprint arXiv:1312.5602*  
[https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602)

## How do I run this?

### GPU setup

These models were trained and run using a commercial Nvidia GPU. Using Tensorflow with a GPU on Windows natively is not currently supported, but there is support for Windows Subsystem for Linux (WSL) along with the appropriate driver setup.

* [Instructions for setting up WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
* [Instructions for setting up Tensorflow's GPU support with WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

You can run these using CPU only, but it's going to be **slow**.

### Required libraries

You can install the required libraries using pip and the requirements.txt file:

    pip install -r ./requirements.txt

### Training the models

Once gym-retro is set up and the game is integrated, you can train the models by running main.py:

    python -m main

Configuration settings can be set either in modelargs.json or on the command line by specifying --arg and following it with an appropriate value. The arguments are listed below:

* **state**: determines which save state to load the game in. Under the integration folder, there are multiple states. The default is **gamestart**, which starts the player at the beginning of the game. Each **level state** (levelx.state) starts at dungeon x and contains only the minimum of what is needed to either get to or beat that dungeon. For both configuration and command line arguments, only use the name of the state, without the file extension.
* **model**: currently set to DQN, as the other models were written with an older version of this code and still need to be updated.
* **game**: set to "Zelda1" - hoping to make this framework more agnostic in the future.
* **num_episodes**: the number of episodes for the model to run.
* **learning_rate**: controls how quickly the model updates its parameters during training.
* **discount_factor**: determines the present value of future rewards, balancing immediate rewards against long-term gains.
* **epsilon**: the probability that the agent will take a random action instead of following its current learned policy, encouraging exploration of the environment.
* **epsilon_decay**: the factor by which epsilon is reduced after each episode. This gradual reduction allows the agent to shift from exploration towards exploitation as it learns.
* **epsilon_min**: the minimum value to which epsilon can decay. Ensures that the agent always retains a small probability of taking random actions, which can help prevent the policy from becoming completely deterministic.
* **max_frames**: the maximum amount of frames for the agent to step through. 

Note that the model will be saved under the directory ./saved_models.

### Loading models

The code, by default, will load the most recent model based on the model parameter. You can force it to start a new model by commenting these lines:

    # load a pre-trained model into the agent.
    load_model_into_agent(agent)

## To-do list 

* Optimize the reward function parameters to better encourage exploration.
* Update the other models to work with the current iteration of the framework.
* Create a function to allow the user to update values (such as giving the agent specific weapons, number of rupees, etc.) defined in a JSON file.
* Create a more robust model load feature and allow user to load model based on arguments.

## Sources

*Some RAM addresses were sourced from:*
* [DataCrystal](https://datacrystal.romhacking.net/wiki/The_Legend_of_Zelda:RAM_map)
* [Zophar's Domain](https://zeldit.zophar.net/hack.html)
* [Trax's Rom Hacking](https://www.bwass.org/romhack/zelda1/zelda1rammap.txt)