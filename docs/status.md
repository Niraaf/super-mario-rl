---
layout: default
title: Status
---

# {{ page.title }}


## Project Summary


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
plan on using incremental randomized level subsets
(have model train on 1-1 and 1-2 until it passes, then on 1-1, 1-2, and 1-3 until it passes, etc)

goal recalibration (compare goals from proposal to what we were able to achieve rn)

etc

## Resources Used
