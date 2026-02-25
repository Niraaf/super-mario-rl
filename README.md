# CS 175: Super Mario Bros RL

This repository contains the RL project for CS 175 (Winter 2026). We are training a PPO agent to play Super Mario Bros using the `gym-super-mario-bros` environment.

## Team
* **Christian Lasam**
* **Jovan Ng**
* **Farin Soriano**

## Setup Instructions

### 1. Create a virtual environment
```bash
python -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```
You have to activate the virtual environment every time you open your IDE. VSCode allows automation of this by selecting venv as the Python interpreter.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## How to Run the Code
### To Train the Agent (CNN)
This script creates the environment, initializes the PPO model, and starts training.
* Models are automatically saved to the models/ folder with a timestamp (e.g., mario_cnn_0207_1230.zip).

```bash
python train_cnn.py
```

### To Watch the Agent Play
This script loads a trained model from models/ and generates a GIF in replays/.
* Usage: python record_gif_cnn.py [model_name]

Example: If your model file is models/mario_cnn_0207_1230.zip:

```bash
python record_gif_cnn.py mario_cnn_0207_1230
```

### To Train the Agent (Phase 3: LSTM + Curriculum)
This script initializes the advanced architecture (CnnLstmPolicy) and utilizes the custom ErrorDrivenCurriculumWrapper and AntiStallWrapper to teach the agent generalized physics across multiple levels.

* To start a fresh training run:

 ```bash
 python train_v3.py
 ```
* To resume training from an existing checkpoint:
 Pass the name of the model file (you do not need to include the models/ folder prefix or the .zip extension).

 ```bash
 python train_v3.py mario_lstm_phase3_0224_1530
 ```
* To Watch the Agent Play (Phase 3: LSTM)
 This script evaluates your Phase 3 LSTM model and records a GIF of its performance. It can test both stochastic (training) and deterministic (evaluation) behaviors.

 ```bash
 Usage: python record_gif_v3.py [model_name]
 ```

Example:

```bash
python record_gif_v3.py mario_lstm_phase3_0224_1530
```