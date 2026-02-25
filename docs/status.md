---
layout: default
title: Status
---

# {{ page.title }}

<iframe width="560" height="315" src="https://www.youtube.com/embed/zPe-sGxUEUc?si=MwBfg0v1prnbnKEC" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Project Summary

Our project trains a reinforcement learning agent to play Super Mario Bros using Proximal Policy Optimization (PPO) with a convolutional neural network (CNN) policy. The agent receives raw game frames from the NES emulator through the `gym-super-mario-bros` OpenAI Gym environment, preprocesses them into stacked grayscale images, and learns a policy that maps visual observations to game actions. The core challenge is teaching the agent to navigate a complex, dynamic environment — avoiding enemies and obstacles, managing momentum, and progressing through levels — using only pixel-level feedback and a shaped reward signal based on position change, time penalties, and death penalties. Our current prototype trains a PPO-CNN agent on World 1-1 using the SIMPLE_MOVEMENT action space, and we are evaluating how far the agent can progress and how consistently it completes the level across training checkpoints.

## Approach

Our core method utilizes PPO paired with a CNN policy to train an agent to navigate the Super Mario Bros environment. We process the visual state using a 4-frame stack, converting frames to grayscale and resizing them to 84x84 pixels. Our action space is restricted to SIMPLE_MOVEMENT to reduce exploration complexity.

To thoroughly explore the problem space, our team split our training into two parallel experimental tracks:

1. Track 1 (Convergence & Fine-Tuning): We hyper-tuned a model exclusively on World 1-1 to study deterministic execution. To stabilize this late-stage model, we dropped the learning rate to 0.00001 and the entropy coefficient to 0.002.

2. Track 2 (Generalization & Reward Shaping): We trained a separate long-horizon agent for over 10 million+ timesteps across randomized environments. For this model, we engineered a custom reward function that heavily incentivized vertical (Y-axis) movement alongside standard rightward progression to encourage jumping and exploration.

## Evaluation

We evaluated our models quantitatively via TensorBoard metrics and qualitatively through deterministic and stochastic deployment.

**Track 1: Peak Performance, State Aliasing, & Domain Shift**
For our 1-1 specialized model, training metrics converged around 1.8M timesteps. Our 2.2M timestep model represents our peak deterministic policy, executing a level clear in 393 frames. Meanwhile, our 2.55M timestep model represents our peak stochastic policy (deterministic=False), achieving a blistering 310-frame clear.

However, evaluating checkpoints beyond 2.2M steps with deterministic=True revealed severe policy degradation. The agent attempted to over-optimize forward momentum to minimize clock penalties at the final staircase, resulting in a sub-pixel wall collision. Due to the 4-frame stack, the complete loss of momentum caused state aliasing—trapping the agent in a paralyzed "run right" loop against a solid collision box. So, when setting deterministic to True, the model would get stuck at the very last staircase before the flag.

Furthermore, when testing our 1-1 Champion model's generalization, we observed a massive vulnerability to visual domain shift. Dropping the agent into World 1-2 resulted in failure within 23 frames (dying to the first Goomba), as the CNN had overfit to the daylight palette and failed to recognize the dark blue tilesets.

**Track 2: Randomized Training & Reward Exploitation**
Our generalized 10M+ timestep model demonstrated mediocre performance across most standard platforming levels, struggling to learn precise gap-jumping and enemy avoidance. However, it performed disproportionately well on water levels (e.g., World 2-2). This success was a direct result of our custom vertical reward shaping combined with swimming physics. Because swimming allows continuous vertical adjustment, the agent aggressively exploited the Y-axis reward to stay near the top of the screen. This allowed it to float over most threats and survive significantly longer, effectively bypassing its lack of fundamental platforming skills.

## Remaining Goals and Challenges
For the remainder of the quarter, we want to be able to create an agent that can somewhat consistently beat all the levels in the game. As of right now, with the ways we have trained the model, it is either a ultra specialist that can't do any other level, or it is bad to mediocre on all the levels. Compared to our original goals, our current goals are pretty similar. We can get models to beat each level individually, the hardest challenge is getting the model to generalize.

One way we plan on doing that is training the model on incremental randomized level subsets. For example, we have the model go until it beats 1-1. After that we do 1-1 and 1-2 until is passes. We continue this pattern until we can get the model to complete all 32 levels with this method. The potential challenges we face are not enough time and catastrophic unlearning. Training it to beat one level takes around 1.7M timesteps, so doing it with this method to try and get it to beat every level would take an incredible amount of time and compute power. Additionally, when it first start on the new level, it has no idea what to do, causing the reward function to crash.

One other way that we plan to train it is to use subproc_vec_env from StableBaselines3 to allow the model to train on multiple environments at the same time. It forks different environments that run in parallel, and the model will train on all of them. It allows the agent to train n environments every step instead of 1, where n is a variable set to determine number of environments. The challenges with this model is that we do not know how the hyperparameters and custom reward function will interact with multiple environments. Additionally, since there are more timesteps per step due to the multiple environments, the graph is not as detailed, making it harder to analyze.

## Resources Used

The following resources were used in the implementation, experimentation, and analysis of our project:

- **[gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)**: The primary Gym environment used to interface with the Super Mario Bros. game. Provided the observation space, action space definitions (including `SIMPLE_MOVEMENT`), and the default reward function based on x-position delta, time, and death.

- **[nes-py](https://github.com/Kautenja/nes-py)**: NES emulator wrapped as a Python Gym environment, used as the backend for `gym-super-mario-bros`.

- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)**: Used for the PPO implementation (`PPO` with `CnnPolicy`). Documentation consulted for hyperparameter defaults, callback setup, and vectorized environment usage.

- **[Gymnasium](https://gymnasium.farama.org/)**: Modern fork of OpenAI Gym, used as the base environment API. Wrappers including `GrayscaleObservation`, `ResizeObservation`, and `FrameStackObservation` were applied for standard Atari-style preprocessing.

- **[Shimmy](https://github.com/Farama-Foundation/Shimmy)**: Used via `GymV21CompatibilityV0` to bridge the older Gym v21 API of `gym-super-mario-bros` with the Gymnasium API.

- **[Pyglet](https://pyglet.org/)**: Required as a rendering backend dependency (pinned to version 1.5.21 for compatibility).

- **[Pillow (PIL)](https://pillow.readthedocs.io/)**: Used for saving recorded gameplay frames as animated GIFs for qualitative evaluation.

- **AI Tool Usage:** Gemini was used throughout the project to assist with debugging and dependency resolution. The gym-super-mario-bros environment has several known compatibility issues stemming from its reliance on the older Gym v21 API, and Gemini was consulted to diagnose and resolve conflicts between gymnasium, gym-super-mario-bros, nes-py, and shimmy. It also helped identify the correct pinned versions for packages like pyglet==1.5.21 and numpy<2.0.0 to ensure a stable training environment. Claude was used minimally to help organize and summarize our updates and accomplishments. No AI tools were used in the design of the reward function or the training pipeline architecture. 