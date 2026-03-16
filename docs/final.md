---
layout: default
title: Final Report
---

# {{ page.title }}
[video]

## Project Summary

[Part of the evaluation will be on how well you are able to
motivate what’s interesting and challenging about the problem, i.e. why it is not trivial, and
why you need AI/ML algorithms to solve it.]

## Approach

Our baseline method utilizes standard PPO with a CNN policy. We process the visual state by converting frames to grayscale and resizing them to 84x84 pixels. For our non-recurrent models, we used a standard 4-frame stack to simulate motion. The action space is strictly limited to SIMPLE_MOVEMENT (7 actions) to reduce exploration complexity. 

To thoroughly explore the problem space, we deployed three distinct experimental training tracks:

**Method 1: The 1-1 Specialist (Convergence & Fine-Tuning)**
We hyper-tuned a model exclusively on World 1-1 to study deterministic execution. To stabilize this late-stage model, we dropped the learning rate to 0.00001 and the entropy coefficient to 0.002.

**Method 2: The Generalist (Randomized Training & Reward Shaping)**
We trained a separate long-horizon agent for over 10 million+ timesteps across randomized environments. For this model, we engineered a custom reward function that heavily incentivized vertical (Y-axis) movement alongside standard rightward progression to encourage jumping and exploration:
`[REWARD FUNCTION DETAILS HERE - prof said we should include exact function]`

**Method 3: Recurrent Policy & Curriculum Learning (CNN + LSTM)**
To address failures observed in the first two methods (state aliasing and catastrophic forgetting), we overhauled the architecture to use `RecurrentPPO` from `sb3-contrib`. 
* **Memory:** We replaced the 4-frame stack with an LSTM network (`n_steps=512`), giving the agent true temporal memory to track its own velocity and momentum.
* **Anti-Stall Wrapper:** Early LSTM models developed a "fear plateau," freezing to avoid death penalties. We implemented a custom `AntiStallWrapper` (120-frame threshold) to violently penalize hesitation and force sprinting.
* **Error-Driven Curriculum Learning:** We built an `ErrorDrivenCurriculumWrapper` that starts the agent exclusively on World 1-1. It evaluates a rolling window of the last 100 lives and requires a strict 80% win rate before unlocking the next level. We utilized a dynamic linear learning rate schedule (starting at 2.5e-4) and an entropy decay callback (0.05 decaying to 0.005) to balance initial exploration with late-stage exploitation.

## Evaluation

We evaluated our models quantitatively via TensorBoard metrics and qualitatively through both deterministic and stochastic deployments.

**Method 1: State Aliasing & Domain Shift**
Our 1-1 specialized model converged around 1.8M timesteps. The stochastic policy achieved a 310-frame clear, while the deterministic policy executed a 393-frame clear. However, evaluating checkpoints beyond 2.2M steps with `deterministic=True` revealed severe policy degradation. The agent attempted to over-optimize forward momentum to minimize clock penalties at the final staircase, resulting in a sub-pixel wall collision. Because the 4-frame stack cannot distinguish "standing still" from "moving right but blocked by a wall," this caused state aliasing—trapping the agent in an infinite "run right" loop. Furthermore, dropping this specialized agent into World 1-2 resulted in failure within 23 frames. The CNN filters had overfit to the bright daylight palette, rendering the agent effectively "blind" to the contrast inversion of the dark underground.

![Specialist on level 1-1](/docs/assets/specialist_1-1.gif)
![Specialist on level 1-2](/docs/assets/specialist_1-2.gif)

**Method 2: Reward Exploitation**
The generalized 10M timestep model demonstrated mediocre performance across standard platforming levels, struggling with precise gap-jumping. However, it performed disproportionately well on water levels (e.g., World 2-2). Because swimming allows continuous vertical adjustment, the agent aggressively exploited our custom Y-axis reward to stay near the top of the screen, bypassing threats entirely. This demonstrated emergent exploitation of reward shaping rather than learned navigation.

![Generalist on random levels](/docs/assets/generalist_random.gif)
![Generalist on level 1-1](/docs/assets/generalist_1-1.gif)

**Method 3: Breaking Local Minimums & The Compute Bottleneck**
The LSTM model successfully resolved the 4-frame stack state aliasing. Around 800k timesteps, the Critic network's explained_variance spiked to 0.96, proving the LSTM had finally calibrated to predict long-term jump trajectories. The agent successfully learned to sprint deep into the level, navigating the final 1-1 staircase that previously trapped Method 1.However, as exploration (entropy) decayed to 0.5%, the algorithm locked into a wide local minimum, determining that safely securing 1,900 points was mathematically safer than risking early death to experiment with the precise flagpole jump. The agent eventually averaged a 25% win rate, peaking at 35% in a 100-game window. Because it never hit the 80% promotion threshold, the curriculum wrapper successfully prevented promotion to 1-2.Crucially, the TensorBoard metrics showed that both the frontier win rate and episodic reward mean were maintaining a steady, upward trajectory when training concluded. The strict 80% promotion threshold simply proved to be a compute bottleneck for the project's timeframe. Recurrent policies require exponentially more training timesteps than standard CNNs - given a longer training horizon and a higher sustained entropy floor to encourage continued exploration of the flagpole jump, the data strongly suggests the LSTM would eventually break the threshold and graduate to future levels. When we manually forced the 12M step LSTM model into World 1-2 for a zero-shot generalization test, it failed instantly, confirming that Out-of-Distribution (OOD) visual domain shift remains the ultimate bottleneck.

![RecurrentPPO reward graph](/docs/assets/lstm_tensorboard.png)
![RecurrentPPO winrate graph](/docs/assets/lstm_winrate.png)
![RecurrentPPO on level 1-1](/docs/assets/lstm_1-1.gif)
![RecurrentPPO on level 1-2](/docs/assets/lstm_1-2.gif)

## Resources Used

The following resources were used in the implementation, experimentation, and analysis of our project:

- **[gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)**: The primary Gym environment used to interface with the Super Mario Bros. game. Provided the observation space, action space definitions (including `SIMPLE_MOVEMENT`), and the default reward function based on x-position delta, time, and death.
- **[nes-py](https://github.com/Kautenja/nes-py)**: NES emulator wrapped as a Python Gym environment, used as the backend for `gym-super-mario-bros`.
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)**: Used for the primary PPO and CNN policy implementations, hyperparameter references, callback setup, and vectorized environment usage (`SubprocVecEnv`).
- **[SB3-Contrib](https://sb3-contrib.readthedocs.io/)**: Used specifically for the `RecurrentPPO` implementation to build the CNN-LSTM architecture and temporal memory tracking in Method 3.
- **[Gymnasium](https://gymnasium.farama.org/)**: Modern fork of OpenAI Gym, used as the base environment API. Wrappers including `GrayscaleObservation`, `ResizeObservation`, and `FrameStackObservation` were applied for standard Atari-style preprocessing.
- **[Shimmy](https://github.com/Farama-Foundation/Shimmy)**: Used via `GymV21CompatibilityV0` to bridge the older Gym v21 API of `gym-super-mario-bros` with the Gymnasium API.
- **[TensorBoard](https://www.tensorflow.org/tensorboard)**: Essential tool used for tracking our training metrics, analyzing policy gradient losses, and visually identifying our agent's local minimums and breakthroughs via `ep_rew_mean` and `explained_variance` graphs.
- **[Pyglet](https://pyglet.org/)**: Required as a rendering backend dependency (pinned to version 1.5.21 for compatibility).
- **[Pillow (PIL)](https://pillow.readthedocs.io/)**: Used for saving recorded gameplay frames as animated GIFs for qualitative evaluation and final presentation videos.
- **AI Tool Usage:** Gemini was used throughout the project to assist with debugging and dependency resolution. The gym-super-mario-bros environment has several known compatibility issues stemming from its reliance on the older Gym v21 API, and Gemini was consulted to diagnose and resolve conflicts between gymnasium, gym-super-mario-bros, nes-py, and shimmy. It also helped identify the correct pinned versions for packages like pyglet==1.5.21 and numpy<2.0.0 to ensure a stable training environment. Claude was used minimally to help organize and summarize our updates and accomplishments. No AI tools were used in the design of the reward function or the training pipeline architecture. 