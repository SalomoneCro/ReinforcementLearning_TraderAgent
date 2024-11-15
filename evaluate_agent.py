from stable_baselines3 import PPO
from agent import Trader
import numpy as np

# Configuración inicial del entorno para evaluación
n_assets = 3  # Número de activos
tickers = ["AAPL", "MSFT", "GOOGL"]  # Tickers de ejemplo
start_date = "2020-05-01"  # Fecha de inicio para evaluación
end_date = "2020-07-01"    # Fecha de fin para evaluación
initial_weights = [0.33, 0.33, 0.34]  # Porcentaje inicial asignado a cada activo
initial_investment = 100000  # Inversión inicial en dólares

# Crear el entorno de evaluación
eval_env = Trader(
    n_assets=n_assets,
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    initial_weights=initial_weights,
    initial_investment=initial_investment,
)

# Cargar el modelo entrenado
model = PPO.load("ppo_trader_model")

# Evaluación
obs = eval_env.reset()
done = False
portfolio_values = []  # Lista para almacenar el valor del portafolio

while not done:
    # Obtener acción del modelo
    action, _ = model.predict(obs, deterministic=True)
    
    # Tomar un paso en el entorno
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    # Almacenar el valor del portafolio actual
    portfolio_value = eval_env.calculate_reward()
    portfolio_values.append(portfolio_value)
    
    # Verificar si el episodio ha terminado
    done = terminated or truncated

# Resultados finales
print(f"Valor final del portafolio: ${portfolio_values[-1]:,.2f}")
print(f"Ganancia porcentual: {((portfolio_values[-1] / initial_investment) - 1) * 100:.2f}%")

# Visualización de resultados
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(portfolio_values, label="Valor del Portafolio")
plt.title("Evolución del Portafolio durante Evaluación")
plt.xlabel("Días de Trading")
plt.ylabel("Valor del Portafolio ($)")
plt.legend()
plt.grid()
plt.show()
