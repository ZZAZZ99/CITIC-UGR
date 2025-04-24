import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp  # Librería para resolver QP

# ----------------------
# 1. Definir los datos bidimensionales
# ----------------------
# En este ejemplo x_data es fijo (todos ceros) y y_data varía.
x_data = np.array([0, 0, 0, 0, 0])
y_data = np.array([0, 1, 2, 3, 4])
n_points = len(x_data)

# ----------------------
# 2. Calcular el parámetro t (distancia acumulada) y normalizarlo a [0, 1]
# ----------------------
t = np.zeros(n_points)
for i in range(1, n_points):
    t[i] = t[i-1] + np.sqrt((x_data[i]-x_data[i-1])**2 + (y_data[i]-y_data[i-1])**2)
t = (t - t.min())/(t.max()-t.min())

# ----------------------
# 3. Definir las funciones base para x(t) e y(t)
# ----------------------
# Estas funciones base permiten expresar la curva resultante mediante un único
# vector de coeficientes lambda. En este ejemplo:
#
# Para x(t) se desea reproducir la función constante cero. Se elige:
#   - x1(t) = 0
#   - x2(t) = t
# Así, x_fit(t) = lambda[0]*0 + lambda[1]*t = lambda[1]*t.
#
# Para y(t) se eligen dos funciones base:
#   - y1(t) = 3*t
#   - y2(t) = 2*t
# De manera que y_fit(t) = lambda[0]*(3*t) + lambda[1]*(2*t).
#
# Notar que, al usar el mismo lambda para ambas coordenadas, se impone que la
# “forma” de la curva esté dada por una única combinación convexa.
A_x = np.vstack([np.zeros_like(t), t]).T       # Dimensión: (n_points, 2)
A_y = np.vstack([3*t, 2*t]).T                    # Dimensión: (n_points, 2)

# ----------------------
# 4. Construir la matriz total y el vector de datos total para el QP
# ----------------------
# Se apilan las ecuaciones para x y para y de forma vertical.
A_total = np.vstack([A_x, A_y])                  # Dimensión: (2*n_points, 2)
b_total = np.concatenate([x_data, y_data])       # Vector de longitud 2*n_points

# Formular el problema QP:
# Queremos minimizar:
#     1/2 * ||A_total * lambda - b_total||^2
# que se escribe en forma estándar con:
P = 2 * A_total.T @ A_total                    # Matriz 2x2
q = -2 * A_total.T @ b_total                   # Vector de longitud 2

# Restricciones:
#  - Constrain: lambda >= 0
G = -np.eye(2)                                 # Para imponer: -lambda <= 0  <=> lambda >= 0
h_vec = np.zeros(2)

#  - Restricción de combinación convexa: lambda[0] + lambda[1] = 1
A_eq = np.ones((1, 2))
b_eq = np.array([1.0])

# Resolver el QP usando, por ejemplo, el solver OSQP.
lambda_shared = solve_qp(P, q, G, h_vec, A_eq, b_eq, solver='osqp')
print("Solución (lambdas compartidas):", lambda_shared)

# ----------------------
# 5. (Opcional) Post-procesar los lambda para redondear pequeñas tolerancias numéricas
# ----------------------
tol = 1e-3
lambda_shared = np.where(np.abs(lambda_shared) < tol, 0, lambda_shared)
lambda_shared = lambda_shared / np.sum(lambda_shared)  # Aseguramos que sumen 1
print("Solución post procesada:", lambda_shared)

# ----------------------
# 6. Calcular el ajuste en una malla fina de t para visualizar la curva
# ----------------------
t_fine = np.linspace(0, 1, 500)

# Evaluar las funciones base en la malla fina:
A_x_fine = np.vstack([np.zeros_like(t_fine), t_fine]).T
A_y_fine = np.vstack([3*t_fine, 2*t_fine]).T

# Construir la curva ajustada usando los mismos lambda para x y para y.
x_fit = A_x_fine @ lambda_shared
y_fit = A_y_fine @ lambda_shared

# ----------------------
# 7. Graficar el resultado
# ----------------------
plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, color='red', label='Datos originales')
plt.plot(x_fit, y_fit, label='Ajuste obtenido (QP)', color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste bidimensional con base común (QP)")
plt.legend()
plt.grid(True)
plt.show()