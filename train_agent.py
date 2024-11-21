import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from agent import Trader
from utils import get_stocks



def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO agent")
    
    parser.add_argument(
        "--model_name",
        "-mn",
        type=str, 
        default="ppo_trader_model_default", 
        help="Name of the model"
    )
    
    parser.add_argument(
        "--timesteps", 
        "-ts",
        type=int, 
        default=1, 
        help="Number of timesteps for training"
    )
    return parser


def main():

    # Parse arguments
    args = init_parser().parse_args()
    model_name = args.model_name

    # Define portdolio
    tickers = ['TSLA', 'GOOGL', 'MELI', 'MSI', 'NVDA']  
    n_assets = len(tickers)
    start_date = "2024-01-01"
    end_date = "2024-06-01"
    initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2] 
    initial_investment = 100
    stock_prices = get_stocks(tickers, start_date, end_date)

    # Create the environment
    env = Trader(
        stock_prices, 
        n_assets, 
        tickers, 
        initial_weights, 
        initial_investment
    )

    vec_env = make_vec_env(lambda: env, n_envs=1)

    # PPO config
    model = PPO(
        "MlpPolicy",  
        vec_env,      
        verbose=1,    
        tensorboard_log="./ppo_trader_tensorboard/",  
    )

    # Train the model
    timesteps = args.timesteps 
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # Save trained model
    if not os.path.exists('./Models'):
        os.makedirs('./Models')
    model.save(f"./Models/{model_name}")


if __name__ == "__main__":
    main()