import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from agent import Trader
from utils import get_stocks
from time import time

# Configurar argparse para recibir parámetros desde la línea de comandos
parser = argparse.ArgumentParser(description="Entrena un modelo PPO para trading")
parser.add_argument(
    "--model_name", 
    type=str, 
    default="ppo_trader_model_default", 
    help="Nombre para guardar el modelo entrenado"
)

# Parsear los argumentos
args = parser.parse_args()
model_name = args.model_name

# Configuración inicial del entorno
tickers = ['TSLA', 'GOOGL', 'MELI', 'MSI', 'NVDA']  
n_assets = len(tickers)
start_date = "2024-01-01"
end_date = "2024-06-01"
initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2] 
initial_investment = 100
stock_prices = get_stocks(tickers, start_date, end_date)

# Crear el entorno de trading
env = Trader(
    stock_prices, 
    n_assets, 
    tickers, 
    initial_weights, 
    initial_investment
)

vec_env = make_vec_env(lambda: env, n_envs=1)

# Configuración y entrenamiento del modelo PPO
model = PPO(
    "MlpPolicy",  
    vec_env,      
    verbose=1,    
    tensorboard_log="./ppo_trader_tensorboard/",  
)

a = time()

# Entrenar el modelo
timesteps = 300000  
model.learn(total_timesteps=timesteps, progress_bar=True)

# Guardar el modelo entrenado
model.save(f"./Models/{model_name}")

print("Entrenamiento completado y modelo guardado.")
print((time() - a) / 60 / 60)
