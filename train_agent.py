from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from agent import Trader
from utils import get_stocks

# Configuración inicial del entorno

tickers = ['TSLA', 'GOOGL', 'MELI', 'MSI', 'NVDA']  # Tickers de ejemplo
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
timesteps = 300000  # Número de pasos de entrenamiento
model.learn(total_timesteps=timesteps, progress_bar=True)

# Guardar el modelo entrenado
model.save("ppo_trader_model2")

print("Entrenamiento completado y modelo guardado.")
