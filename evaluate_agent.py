import argparse
from stable_baselines3 import PPO
from agent import Trader
from utils import get_stocks, save_plots, print_metrics



def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena un modelo PPO para trading")
    
    parser.add_argument(
        "--model_name",
        "-mn",
        type=str, 
        default="ppo_trader_model_default", 
        help="Nombre del modelo entrenado"
    )
    
    parser.add_argument(
        "--n_simulations", 
        "-n_sim",
        type=int, 
        default=100, 
        help="Numero de simulaciones para la evaluación"
    )
    return parser


def main():
    args = init_parser().parse_args()

    # Define portdolio
    tickers = ['TSLA', 'GOOGL', 'MELI', 'MSI', 'NVDA']
    n_assets = len(tickers)
    start_date = "2024-06-01"
    end_date = "2024-11-15"
    initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    initial_investment = 100
    stock_prices = get_stocks(tickers, start_date, end_date)

    # Create the environment
    eval_env = Trader(
        stock_prices,
        n_assets,
        tickers,
        initial_weights,
        initial_investment
    )

    # Load the trained agent
    model = PPO.load(f'./Models/{args.model_name}')

    # SIMULATIONS

    all_simulations = []
    n_simulations = args.n_simulations
    for i in range(n_simulations):
        obs, _ = eval_env.reset()
        done = False

        portfolio_values = [initial_investment]
        shares_over_time = [eval_env.shares.copy()]
        dates = [eval_env.start_date]

        while not done:
            action, _ = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, info = eval_env.step(action)

            done = terminated or truncated

            portfolio_values.append(eval_env.calculate_reward())
            shares_over_time.append(eval_env.shares.copy())
            dates.append(eval_env.current_date)

        all_simulations.append({
            "portfolio_values": portfolio_values,
            "shares_over_time": shares_over_time,
            "final_value": portfolio_values[-1]
        })

    # RESULTS

    save_plots(all_simulations, dates, tickers)
    print_metrics(all_simulations, initial_investment)


if __name__ == "__main__":
    main()