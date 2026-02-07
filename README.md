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

## To Watch the Agent Play
This script loads a trained model from models/ and generates a GIF in replays/.
* Usage: python record_cnn.py [model_name]

Example: If your model file is models/mario_cnn_0207_1230.zip:

```bash
python record_cnn.py mario_cnn_0207_1230
```