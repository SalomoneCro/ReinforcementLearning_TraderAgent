import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from agent import Trader
from utils import get_stocks

# Configuración inicial del entorno
n_assets = 3  # Número de activos
tickers = ["AAPL", "MSFT", "GOOGL"]  # Tickers de ejemplo
start_date = "2023-01-01"
end_date = "2023-02-01"
initial_weights = [0.33, 0.33, 0.34]  # Porcentaje inicial asignado a cada activo
initial_investment = 100000  # Inversión inicial en dólares
stock_prices = get_stocks(tickers, start_date, end_date)

# Crear el entorno de trading
env = Trader(
    stock_prices, 
    n_assets, 
    tickers, 
    initial_weights, 
    initial_investment
)

# Convertir el entorno para que funcione en paralelo con Stable-Baselines
vec_env = make_vec_env(lambda: env, n_envs=1)

# Configuración y entrenamiento del modelo PPO
model = PPO(
    "MlpPolicy",  # Política de red neuronal (MLP)
    vec_env,      # Entorno
    verbose=1,    # Nivel de detalle en la consola
    tensorboard_log="./ppo_trader_tensorboard/",  # Carpeta para TensorBoard
)

# Entrenar el modelo
timesteps = 10  # Número de pasos de entrenamiento
model.learn(total_timesteps=timesteps, progress_bar=True)

# Guardar el modelo entrenado
model.save("ppo_trader_model")

print("Entrenamiento completado y modelo guardado.")