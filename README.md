# CS 175: Super Mario Bros RL

This repository contains the RL project for CS 175 (Winter 2026). We are training a PPO agent to play Super Mario Bros using the gym-super-mario-bros environment.

## Team
* Christian Lasam
* Jovan Ng
* Farin Soriano

## Setup Instructions
### 1. Create a virtual environment
python -m venv venv

Mac/Linux: source venv/bin/activate
On Windows: venv\Scripts\activate

### 2. Install dependencies
pip install -r requirements.txt

## How to Run the Code

### To Train the Agent (CNN)
This script creates the environment, initializes the PPO model, and starts training.

python train_cnn.py 

### To Watch the Agent Play
This script loads the trained model and generates a gif of the result. Make sure to rename file_name to the actual zip file.

python record_cnn.py