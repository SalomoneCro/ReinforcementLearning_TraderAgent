# Trading Agent with PPO (Proximal Policy Optimization)

This project implements a trading agent using the PPO (Proximal Policy Optimization) algorithm to weekly make buy and sell decisions on a portfolio with multiple assets.

## Project Structure

The project consists of the following modules:

- `agent.py`: Contains the `Trader` class that defines the trading environment, including the observation and action spaces, portfolio rebalancing logic, and reward calculation.
- `spaces.py`: Defines the custom observation and action spaces for the environment.
- `train_agent.py`: A script to train the agent using PPO from the `Stable Baselines3` library.
- `evaluate_agent.py`: A script to evaluate the performance of the trained agent through simulations and generate performance metrics and plots.
- `utils.py`: Contains utility functions for fetching stock data, generating price observations, and other related tasks.

## Requirements

You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```
## How to run


To train an agent you can run the following command:

```bash
python3 train_agent.py --model_name Agent1 --timesteps 100000
```
To evaluate an agent you can run the following command:

```bash
python3 evaluate_agent.py --model_name Agent1 --n_simulations 100
```

To define a different portfolio, training dates, evaluation dates and initial allocation it is needed to change the lines that define that on the main functions of both train_agent.py and evaluate_agent.py
