import numpy as np
from stable_baselines3 import PPO
from agent import Trader
from utils import get_stocks

# Configuración inicial del entorno para evaluación
tickers = ['TSLA', 'GOOGL', 'MELI', 'MSI', 'NVDA']  # Tickers de ejemplo
n_assets = len(tickers)
start_date = "2024-06-01"
end_date = "2024-09-01"
initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Porcentaje inicial asignado a cada activo
initial_investment = 100 # Inversión inicial en dólares
stock_prices = get_stocks(tickers, start_date, end_date)

eval_env = Trader(
    stock_prices, 
    n_assets, 
    tickers, 
    initial_weights, 
    initial_investment
)

# Cargar el modelo entrenado
model = PPO.load("ppo_trader_model2")

# Lista para almacenar resultados de cada simulación
all_portfolio_values = []

# Ejecutar 100 simulaciones
n_simulations = 100
for i in range(n_simulations):
    # Inicializar el entorno y el estado
    obs, _ = eval_env.reset()
    done = False
    portfolio_values = []  # Lista para almacenar el valor del portafolio de esta simulación
    
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
    
    # Almacenar el valor final de la simulación
    all_portfolio_values.append(portfolio_values[-1])

# Calcular estadísticas de las simulaciones
min_value = np.min(all_portfolio_values)
max_value = np.max(all_portfolio_values)
avg_value = np.mean(all_portfolio_values)
std_value = np.std(all_portfolio_values)

# Calcular la ganancia promedio y porcentual
avg_percentage_return = ((avg_value / initial_investment) - 1) * 100
min_percentage_return = ((min_value / initial_investment) - 1) * 100
max_percentage_return = ((max_value / initial_investment) - 1) * 100

# Mostrar los resultados finales
print('\n\n\n')
print(f"Resultados de 100 simulaciones:")
print(f"Valor final mínimo del portafolio: ${min_value:,.2f} ({min_percentage_return:.2f}%)")
print(f"Valor final máximo del portafolio: ${max_value:,.2f} ({max_percentage_return:.2f}%)")
print(f"Valor final promedio del portafolio: ${avg_value:,.2f} ({avg_percentage_return:.2f}%)")
print(f"Desviación estándar de los valores finales del portafolio: ${std_value:,.2f}")
