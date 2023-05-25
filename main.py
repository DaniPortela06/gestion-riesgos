import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Paso 1: Seleccionar una muestra de activos de mercados emergentes
# Supongamos que tenemos una matriz de rendimientos de activos
returns = pd.read_csv('datos_rendimientos.csv', index_col=0)

# Paso 2: Realizar análisis multivariado para medir la correlación entre los activos
correlation_matrix = returns.corr()

# Paso 3: Utilizar teoría de la cartera de Markowitz para construir carteras eficientes


def calculate_portfolio(weights, returns):
    portfolio_return = np.sum(returns.mean(axis=0) * weights) * 252
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility


def minimize_volatility(weights):
    return calculate_portfolio(weights, returns)[1]

# Definir las restricciones para la optimización


def constraints(weights):
    return np.sum(weights) - 1  # Suma de pesos igual a 1


# Encontrar la cartera de mínima volatilidad
n_assets = returns.shape[1]
# Asignar pesos iniciales uniformes
initial_weights = np.ones(n_assets) / n_assets
bounds = [(0, 1) for _ in range(n_assets)]  # Límites de los pesos entre 0 y 1

result = minimize(minimize_volatility, initial_weights, method='SLSQP',
                  bounds=bounds, constraints={'type': 'eq', 'fun': constraints})
optimal_weights = result.x

# Paso 4: Utilizar técnicas de optimización para ajustar las carteras y reducir aún más el riesgo

# ... Aquí podrías agregar más pasos de optimización según tus necesidades ...

# Imprimir los pesos óptimos
print("Pesos óptimos de la cartera:")
for i in range(n_assets):
    asset = returns.columns[i]
    weight = optimal_weights[i]
    print(f"{asset}: {weight}")

# Calcular el rendimiento y la volatilidad de la cartera óptima
portfolio_return, portfolio_volatility = calculate_portfolio(
    optimal_weights, returns)
print("\nCartera óptima:")
print(f"Rendimiento anual esperado: {portfolio_return:.2%}")
print(f"Volatilidad anual: {portfolio_volatility:.2%}")
