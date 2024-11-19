import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import pandas as pd
from agent import Trader

# Cargar datos de precios históricos
from utils import get_stocks

# Parámetros iniciales
TICKERS = ["AAPL", "MSFT", "GOOGL"]  # Lista de acciones
START_DATE = "2024-01-01"
END_DATE = "2024-02-04"
INITIAL_WEIGHTS = [0.4, 0.3, 0.3]
INITIAL_INVESTMENT = 100000  # Dinero inicial para invertir

# Obtener datos históricos de precios
stock_prices = get_stocks(TICKERS, START_DATE, END_DATE)

# Crear el entorno personalizado
env = Trader(
    stock_prices=stock_prices,
    n_assets=len(TICKERS),
    tickers=TICKERS,
    initial_weights=INITIAL_WEIGHTS,
    initial_investment=INITIAL_INVESTMENT
)

# Envolver el entorno en un vectorizado (opcional, para estabilidad de PPO)
vec_env = make_vec_env(lambda: env, n_envs=1)

# Inicializar el modelo PPO
model = PPO(
    policy="MlpPolicy",        # Política basada en redes neuronales
    env=vec_env,               # Entorno vectorizado
    verbose=1,                 # Nivel de salida (1 = detallado)
    tensorboard_log="./ppo_trader_tensorboard/"  # Carpeta para visualizar logs en TensorBoard
)

# Entrenar el modelo
model.learn(total_timesteps=20)

# Guardar el modelo
model.save("ppo_trader_model")

# Evaluar el modelo entrenado
obs = vec_env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, truncated, info = vec_env.step(action)
    print(f"Step {i + 1} | Reward: {rewards}")

# Cargar modelo guardado (opcional)
# loaded_model = PPO.load("ppo_trader_model", env=vec_env)
