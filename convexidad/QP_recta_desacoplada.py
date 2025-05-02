import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp  # Librería para resolver QP

# ----------------------
# 1. Definir los datos (bidimensional)
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
    t[i] = t[i-1] + np.sqrt((x_data[i] - x_data[i-1])**2 + (y_data[i] - y_data[i-1])**2)
t = (t - t.min()) / (t.max() - t.min())

# ----------------------
# 3. Definir las funciones base para x(t) y para y(t)
# ----------------------
# Para x(t), queremos poder reproducir la función cero. Por ello incluimos:
#   x1(t) = 0        (función constante cero)
#   x2(t) = t        (por ejemplo, cualquier otra función base, aunque en este caso no será utilizada)
#
# Así, la combinación será:
#   x_fit(t) = λ1 * 0 + λ2 * t = λ2 * t
# Con la restricción λ1 + λ2 = 1, para obtener x_fit(t) = 0 se deberá tener λ2 = 0.
A_x = np.vstack([np.zeros_like(t), t]).T   # Matriz de funciones base para x, dimensión (n_points, 2)

# Para y(t) usamos dos funciones base, por ejemplo:
#   y1(t) = 3 * t
#   y2(t) = 2 * t
A_y = np.vstack([3*t, 2*t]).T   # Matriz de funciones base para y, dimensión (n_points, 2)

# ----------------------
# 4. Formular y resolver el problema QP para cada coordenada
# ----------------------
def solve_for_coordinate(A, b):
    # Minimiza 1/2 * ||A * λ - b||^2 = 1/2 * λ^T * P * λ + q^T * λ
    P = 2 * A.T @ A       # Matriz (2x2)
    q = -2 * A.T @ b      # Vector (2,)
    # Restricción: λ_i >= 0 (Gx <= h con G = -I)
    G = -np.eye(2)
    h_vec = np.zeros(2)
    # Restricción de combinación convexa: λ1 + λ2 = 1
    A_eq = np.ones((1, 2))
    b_eq = np.array([1.0])
    # Resolver el QP
    lambda_opt = solve_qp(P, q, G, h_vec, A_eq, b_eq, solver = 'highs')
    return lambda_opt

# Resolver QP para x(t) y para y(t)
lambda_x = solve_for_coordinate(A_x, x_data)
lambda_y = solve_for_coordinate(A_y, y_data)
print("Solución para x (lambdas):", lambda_x)
print("Solución para y (lambdas):", lambda_y)

# Definir un umbral
tol = 1e-7

# Redondear valores muy pequeños a cero para lambda_x y lambda_y
lambda_x = np.where(np.abs(lambda_x) < tol, 0, lambda_x)
lambda_y = np.where(np.abs(lambda_y) < tol, 0, lambda_y)

# Reescalar para asegurar que sumen 1 (por si la suma se vio afectada)
lambda_x = lambda_x / np.sum(lambda_x)
lambda_y = lambda_y / np.sum(lambda_y)

print("lambda_x post procesado:", lambda_x)
print("lambda_y post procesado:", lambda_y)


# ----------------------
# 5. Calcular el ajuste en una malla fina de t para visualizar la curva
# ----------------------
t_fine = np.linspace(0, 1, 500)
# Evaluar las funciones base en la malla fina
A_x_fine = np.vstack([np.zeros_like(t_fine), t_fine]).T
A_y_fine = np.vstack([3 * t_fine, 2 * t_fine]).T

# Construir las funciones ajustadas
x_fit = A_x_fine @ lambda_x
y_fit = A_y_fine @ lambda_y

# ----------------------
# 6. Graficar el resultado
# ----------------------
plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, color='red', label='Datos originales')
plt.plot(x_fit, y_fit, label='Ajuste obtenido (QP)', color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste bidimensional: x(t) e y(t) mediante combinación convexa y QP")
plt.legend()
plt.grid(True)
plt.show()
