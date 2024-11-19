import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from agent import Trader
from utils import get_stocks

plt.style.use('dark_background')

# Configuración inicial del entorno para evaluación
tickers = ['TSLA', 'GOOGL', 'MELI', 'MSI', 'NVDA']  
n_assets = len(tickers)
start_date = "2024-06-01"
end_date = "2024-11-01"
initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  
initial_investment = 100
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

# SIMULACIONES

all_simulations = []
n_simulations = 100
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

# RESULTADOS

best_simulation = max(all_simulations, key=lambda sim: sim["final_value"])
best_portfolio_values = best_simulation["portfolio_values"]
best_shares_over_time = np.array(best_simulation["shares_over_time"])  # Convertir a NumPy array


# Graficar el valor del portafolio a través del tiempo
plt.figure(figsize=(12, 6))
plt.plot(dates, best_portfolio_values, label="Portfolio Value")
plt.title("Valor del Portafolio a través del Tiempo (Mejor Simulación)")
plt.xlabel("Fecha")
plt.ylabel("Valor del Portafolio ($)")
plt.xticks(rotation=45)
plt.grid(linewidth=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('PortfolioPerformance')


# Graficar la cantidad de acciones en función del tiempo
plt.figure(figsize=(12, 6))
for i, ticker in enumerate(tickers):
    plt.plot(dates, best_shares_over_time[:, i], label=ticker)

plt.title("Cantidad de Acciones en Función del Tiempo (Mejor Simulación)")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de Acciones")
plt.xticks(rotation=45)
plt.grid(linewidth=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Shares')


# Métricas de las simulaciones
final_values = [sim["final_value"] for sim in all_simulations]
max_gain = max(final_values)
min_gain = min(final_values)
mean_gain = np.mean(final_values)
quartiles = np.percentile(final_values, [25, 50, 75])

print("\nMétricas de las simulaciones:")
print(f"Máxima ganancia: ${max_gain:,.2f}")
print(f"Mínima ganancia: ${min_gain:,.2f}")
print(f"Ganancia promedio: ${mean_gain:,.2f}")
print(f"Cuartiles (25%, 50%, 75%): {quartiles}")


std_gain = np.std(final_values)
median_gain = np.median(final_values)
loss_probability = np.mean(np.array(final_values) < initial_investment) * 100
var_95 = np.percentile(final_values, 5)

print("\nMétricas adicionales:")
print(f"Desviación estándar de las ganancias: ${std_gain:,.2f}")
print(f"Mediana de las ganancias: ${median_gain:,.2f}")
print(f"Probabilidad de pérdida: {loss_probability:.2f}%")
print(f"Valor en Riesgo (VaR) al 95%: ${var_95:,.2f}")
