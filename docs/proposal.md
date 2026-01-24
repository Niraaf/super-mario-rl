---
layout: default
title:  Proposal
---

# {{ page.title }}


## Summary of the Project (20 points)
In a paragraph or so, describe the main idea behind your project. Focus on the problem setup, not the solution, i.e. what is your overall goal? If your project is application-driven, what is the application, i.e. the task or behavior that you want the learner / agent to have? If your project is method-driven, what is
the method aiming to achieve and what is an example baseline method? At the very least, you should have a sentence that clearly explains the input/output semantics of your project, i.e. what information will it take as input, and what will it produce.

For our project, we want to user reinforcement learning to create an agent that can play and beat Super Mario Brothers. Super Mario Brothers is a platform game released in 1985 that tasks the player with navigating through obstacles and enemies, as well as managing power ups to beat a series of levels. The controls are up, down, left, right, jump, or run (or attack depending on the powerup). Multiple inputs can be done at once, except for directional inputs which are mutually exclusive. We want our agent to be able to navigate through the game and minimize deaths. Using an OpenAI gym environment on the nes-py emulator, the agent will receive gameplay frames. When it receives the frame, our agent will use the policy it has learned to output the best move at that frame.

## Project goals (20 points)
In a sentence or two each (optionally in a bulleted list), outline 3 levels of goals for the project. Each
level can consist of a single task or behavior that you aim for the agent to do, or a single question you
aim to answer; or if not single then at most a couple related ones. The levels are:

* **Minimum goal:** your most basic effort to get something done and working. If you only do this,
                    you’ll have positive learning experience that’s worth your time.
  At minimum, we plan to train a reinforcement learning agent that can reliably and repeatedly complete all 32 original levels of the game using the standard OpenAI Gym Super Mario Bros. Environment. This includes setting up the environment, preprocessing visual input, defining the action space, and achieveing consistent level completion with a learned policy.

* **Realistic goal:** your good and reasonable effort to do something cool and interesting. If you do
                    all of this, you’ll have a solid project you can be really happy about.
  We plan to extend the agent to handle random variation across levels. We will do this by evaluating it on randomized sequences of subsets of levels provided in the Gym environment. Our agents should demonstrate learning capabilities and be able to complete different types of Mario levels rather than memorizing fixed levels.
  
* **Moonshot goal:** your best effort and dream result, if everything happens to go really well. This
                    may not be realistic, although you could be pleasantly surprised, and anyway the point of this is a
                    “compass” of where the really amazing stuff could be.
Finally, if possible, we'd like to design and modify custom Mario levels and evaluate whether the trained agent can generalize its learned behavior to these new, unseen levels. That means whether the agent has learned general navigation and control strategies rather than overfitting to the game's original levels. This can be accomplished by letting the agent run on new or randomized levels, created manually or through random generation, and observing the number of attempts it needs to complete the level.

## AI/ML Algorithms (10 points)
In a single sentence, name the AI/ML algorithm(s) and method(s) you anticipate using for your project.
It does not have to be a detailed description of any algorithm, even just the name of a class of methods
is sufficient. Examples of this include “planning with dynamic programming”, “reinforcement learning
with neural function approximator”, “learning from demonstrations”, “min-max tree search with
pruning”, and so on. You can take a guess at properties of the method that you’ll prefer using, such as
model-free vs. model-based, on-policy vs. off-policy, etc. You won’t lose points for suggesting an
incorrect algorithm or method, and you’ll be able to switch to different ones at any point that it makes
sense later in the project.

We plan to use reinforcement learning through Proximal Policy Optimization.

## Evaluation Plan (25 points)
As described in class, mention how you will evaluate the success of your project. In a paragraph,
focus on the quantitative evaluation: what experiments you run in order to evaluate, what the metrics
are, what the baselines are (naïve approaches that you’ll try first), by how much you estimate that
your approach could improve the metric (just a ballpark), etc. In another paragraph, describe what
qualitative analysis you will show to verify the project works, such as the debugging, sanity, and toy
cases for the approach, how you will visualize the results (internal and/or external) of the algorithm to
verify it works, what qualitative properties you expect of a successful result, etc.

## AI Tool Usage (5 points)
AI tools are becoming good at some things, not yet at others. Using them for what they’re good for and
watching out for where they fall short is probably something you’re already doing and may choose to
do during your project. Our policy in this course is to allow any use of such tools for your projects.
The only requirement is that you record and report all aspects of the project in which you used AI tools.
Please indicate this clearly in a separate section of your proposal, or otherwise indicate that no AI tools
were used.
You will not lose points in any part of this course for using AI tools smartly, but you may lose points
if you choose to blindly rely on subpar output of such tools — and be warned, they often fail in
ways ranging from obvious to subtle, and their failures may be a time sink to identify. In addition to
“hallucinations”, two high-level usage patterns to be careful about are: The Takeover — you contribute
nothing substantial to the project, at which point ask yourself, if AI can really do your entire job for
you, what you are gaining by pretending otherwise; and The Reversal — you ask the AI for detailed
instructions what to do and your only contribution is to then go and do it, in which case it’s no longer
your AI tool, but rather you’re its human tool. My take: at this point in time, the concern is not that the
above becomes true but rather that you perform your project as if it is.
