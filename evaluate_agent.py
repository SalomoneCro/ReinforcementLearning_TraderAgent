from stable_baselines3 import PPO
from agent import Trader
from utils import get_stocks

# Configuración inicial del entorno para evaluación
  # Número de activos
tickers = ['TSLA', 'GOOGL', 'MELI', 'MSI', 'NVDA']  # Tickers de ejemplo
n_assets = len(tickers)
start_date = "2024-06-01"
end_date = "2024-09-01"
initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Porcentaje inicial asignado a cada activo
initial_investment = 100 # Inversión inicial en dólares
stock_prices = get_stocks(tickers, start_date, end_date)

# Crear el entorno de eavluación
eval_env = Trader(
    stock_prices, 
    n_assets, 
    tickers, 
    initial_weights, 
    initial_investment
)

# Cargar el modelo entrenado
model = PPO.load("ppo_trader_model2")

# Evaluación
obs, _ = eval_env.reset()
done = False
portfolio_values = []  # Lista para almacenar el valor del portafolio

while not done:
    # Obtener acción del modelo
    action, _ = model.predict(obs, deterministic=False)
    
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
plt.savefig('PortfolioPerformance')