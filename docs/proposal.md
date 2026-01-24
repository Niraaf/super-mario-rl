---
layout: default
title:  Proposal
---

# {{ page.title }}


## Summary of the Project

For our project, we want to use reinforcement learning to create an agent that can play and beat Super Mario Brothers. Super Mario Brothers is a platform game released in 1985 that tasks the player with navigating through obstacles and enemies, as well as managing power ups to beat a series of levels. The controls are up, down, left, right, jump, or run (or attack depending on the powerup). Multiple inputs can be done at once, except for directional inputs which are mutually exclusive. We want our agent to be able to navigate through the game and minimize deaths. Using an OpenAI gym environment on the nes-py emulator, the agent will receive gameplay frames. When it receives the frame, our agent will use the policy it has learned to output the best move at that frame.

## Project goals

* **Minimum goal:**
  At minimum, we plan to train a reinforcement learning agent that can reliably and repeatedly complete all 32 original levels of the game using the standard OpenAI Gym Super Mario Bros. Environment. This includes setting up the environment, preprocessing visual input, defining the action space, and achieving consistent level completion with a learned policy.

* **Realistic goal:**
  We plan to extend the agent to handle random variation across levels. We will do this by evaluating it on randomized sequences of subsets of levels provided in the Gym environment. Our agents should demonstrate learning capabilities and be able to complete different types of Mario levels rather than memorizing fixed levels.
  
* **Moonshot goal:**
    Finally, if possible, we'd like to design and modify custom Mario levels and evaluate whether the trained agent can generalize its learned behavior to these new, unseen levels. That means whether the agent has learned general navigation and control strategies rather than overfitting to the game's original levels. This can be accomplished by letting the agent run on new or randomized levels, created manually or through random generation, and observing the number of attempts it needs to complete the level.

## AI/ML Algorithms

We plan to use on-policy reinforcement learning through Proximal Policy Optimization to allow our agent to make decisions.

## Evaluation Plan

Quantatively, to analyze the success of our agent, we can use the deaths positions and farthest reached levels as metrics. We will reward the agent when making it farther in the level, and punish it for not moving, moving backwards, or dying. We can start with a naïve approach of heavy reward for moving forward, and a heavy punishment for moving backward to try and incentivize the agent to quickly beat the level. By altering the reward function, we can allow the agent to be smarter and sometimes move backward in order to pass certain obstacles, which will allow the agent to progress further in the level.

Qualitatively, we can watch the agent and see whether it is making smart decisions in order to beat the levels. The OpenAI gym environment gives the agent 3 lives from the start to beat the game, or 1 life to try and beat a specific level. We can start off by testing the agent on each individual level, and tracking success by seeing how far in the level it can make it in one life. After that, we would move on to seeing how many levels the agent could beat in a normal run of the game. For successful runs, the agent will be able to navigate through the obstacles with a reasonable and time-efficient approach without dying.

## AI Tool Usage

In this proposal, no AI tools were used. For our agent, we will be using the OpenAI Gym for Super Mario Brothers.