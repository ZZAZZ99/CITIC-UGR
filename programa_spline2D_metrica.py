import numpy as np
import matplotlib.pyplot as plt

# Definir los puntos en el plano (x, y)
x = np.array([0, 1, 2, 1, 0])  # Coordenadas x
y = np.array([0, 1, 2, 3, 2])  # Coordenadas y

n = len(x) - 1  # Número de intervalos (n=4 en este caso)

# Paso 1: Definir el parámetro t (podemos usar la distancia acumulada)
t = np.zeros(len(x))

# Escoger la métrica a utilizar para definir el parámetro t

p = input("Indica el orden p de la métrica a utilizar (1, 2, ..., inf): ")

p = float(p)

if p == 1:
    
    for i in range(1, len(x)):
        t[i] = t[i-1] + abs(x[i] - x[i-1]) + abs(y[i] - y[i-1])  # Distancia Manhattan acumulada

elif p == 2:

    for i in range(1, len(x)):
        t[i] = t[i-1] + np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)  # Distancia euclidiana acumulada

elif p > 2:
    for i in range(1, len(x)):
        t[i] = t[i-1] + (abs(x[i] - x[i-1])**p + abs(y[i] - y[i-1])**p)**(1/p)  # Distancia p acumulada

elif p == math.inf:
    for i in range(1, len(x)):
        t[i] = t[i-1] + max(abs(x[i] - x[i-1]), abs(y[i] - y[i-1])) # Distancia de Chebyshev acumulada

else:
        raise ValueError("p debe ser un número mayor o igual a 1 o infinito (math.inf)")


h = np.diff(t)  # Diferencias entre valores de t

# Paso 2: Construir el sistema de ecuaciones para M (segunda derivada)
A = np.zeros((n-1, n-1))  # Matriz de coeficientes
bx = np.zeros(n-1)  # Vector de términos independientes para x
by = np.zeros(n-1)  # Vector de términos independientes para y

for i in range(1, n):
    A[i-1, i-1] = 2 * (h[i-1] + h[i])  # Diagonal principal
    if i-1 > 0:
        A[i-1, i-2] = h[i-1]  # Subdiagonal
    if i < n-1:
        A[i-1, i] = h[i]  # Superdiagonal

    bx[i-1] = (6/h[i]) * (x[i+1] - x[i]) - (6/h[i-1]) * (x[i] - x[i-1])
    by[i-1] = (6/h[i]) * (y[i+1] - y[i]) - (6/h[i-1]) * (y[i] - y[i-1])

# Paso 3: Resolver el sistema para obtener M_x y M_y
Mx = np.zeros(n+1)  # Segunda derivada de x(t)
My = np.zeros(n+1)  # Segunda derivada de y(t)

Mx[1:n] = np.linalg.solve(A, bx)  # Resolver para x
My[1:n] = np.linalg.solve(A, by)  # Resolver para y

# Paso 4: Calcular coeficientes a, b, c, d para cada subintervalo
ax = (Mx[1:] - Mx[:-1]) / (6 * h)
bx = Mx[:-1] / 2
cx = (x[1:] - x[:-1]) / h - (2 * Mx[:-1] + Mx[1:]) * h / 6
dx = x[:-1]

ay = (My[1:] - My[:-1]) / (6 * h)
by = My[:-1] / 2
cy = (y[1:] - y[:-1]) / h - (2 * My[:-1] + My[1:]) * h / 6
dy = y[:-1]

# Paso 5: Evaluar la interpolación en puntos intermedios de t
t_fino = np.linspace(t[0], t[-1], 10000)  # Puntos interpolados, el tercer valor controla la suavidad de la curva
x_fino = np.zeros_like(t_fino)
y_fino = np.zeros_like(t_fino)

for i in range(n):
    mask = (t_fino >= t[i]) & (t_fino <= t[i+1])  # Seleccionamos los puntos en el intervalo
    ti = t_fino[mask] - t[i]  # Convertimos t a coordenadas locales
    x_fino[mask] = ax[i] * ti**3 + bx[i] * ti**2 + cx[i] * ti + dx[i]
    y_fino[mask] = ay[i] * ti**3 + by[i] * ti**2 + cy[i] * ti + dy[i]

# Paso 6: Graficar la curva interpolada
plt.plot(x_fino, y_fino, label="Spline Cúbico Paramétrico", color="blue")
plt.scatter(x, y, color="red", label="Puntos dados")  # Dibujar puntos originales
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Interpolación con Spline Cúbico con norma p = {p}")
plt.legend()
plt.grid(True)
plt.show()
