import numpy as np
import matplotlib.pyplot as plt

# Definir los puntos (modificar manualmente según el problema)
x = np.array([0, 1, 2, 3])  # Coordenadas x
y = np.array([1, 2, 0, 2])  # Coordenadas y

n = len(x) - 1  # Número de intervalos (n=3 en este caso)

# Paso 1: Calcular los valores hi
h = np.diff(x)  # h[i] = x[i+1] - x[i]

# Paso 2: Construir el sistema de ecuaciones para las segundas derivadas (M)
A = np.zeros((n-1, n-1))  # Matriz del sistema
b = np.zeros(n-1)  # Vector de términos independientes

for i in range(1, n):  # Iteramos sobre los puntos internos (sin los extremos)
    A[i-1, i-1] = 2 * (h[i-1] + h[i])  # Elementos diagonales
    if i-1 > 0:
        A[i-1, i-2] = h[i-1]  # Elementos fuera de la diagonal
    if i < n-1:
        A[i-1, i] = h[i]  # Elementos fuera de la diagonal
    b[i-1] = (6 / h[i]) * (y[i+1] - y[i]) - (6 / h[i-1]) * (y[i] - y[i-1])

# Resolver el sistema de ecuaciones lineales para obtener las segundas derivadas (M)
M = np.zeros(n+1)  # Inicializamos M con ceros en los extremos (condición de spline natural)
M[1:n] = np.linalg.solve(A, b)  # Resolver el sistema

# Paso 3: Calcular los coeficientes a, b, c, d para cada subintervalo
a = (M[1:] - M[:-1]) / (6 * h)
b = M[:-1] / 2
c = (y[1:] - y[:-1]) / h - (2*M[:-1] + M[1:]) * h / 6
d = y[:-1]

# Paso 4: Evaluar los polinomios y graficar
x_fino = np.linspace(x[0], x[-1], 10000)  # Puntos interpolados, el tercer valor controla la suavidad de la curva
y_fino = np.zeros_like(x_fino)

# Evaluar el spline en cada subintervalo
for i in range(n):
    mask = (x_fino >= x[i]) & (x_fino <= x[i+1])  # Seleccionamos el intervalo correspondiente
    xi = x_fino[mask] - x[i]
    y_fino[mask] = a[i] * xi**3 + b[i] * xi**2 + c[i] * xi + d[i]

# Graficar los puntos originales y la interpolación
plt.scatter(x, y, color='red', label='Puntos dados')  # Puntos originales
plt.plot(x_fino, y_fino, label='Spline cúbico', linestyle='-', color='blue')  # Curva del spline
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolación con Spline Cúbico (Método Paso a Paso)')
plt.grid(True)
plt.show()

