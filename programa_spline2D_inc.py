import numpy as np
import matplotlib.pyplot as plt

# Definir los puntos en el plano (x, y)
x = np.array([0, 1, 2, 1, 0])  # Coordenadas x
y = np.array([0, 1, 2, 3, 2])  # Coordenadas y

# Definir incertidumbre direccional para cada punto:
# Cada valor de sigma_dir indica la magnitud del error y cada valor de theta_dir (en radianes) la orientación
sigma_dir = np.array([0.15, 0.10, 0.20, 0.15, 0.10])
theta_dir = np.array([np.pi/6, np.pi/4, np.pi/3, np.pi/8, np.pi/10])

n = len(x) - 1  # Número de intervalos

# Paso 1: Definir el parámetro t (usando la distancia acumulada)
t = np.zeros(len(x))
for i in range(1, len(x)):
    t[i] = t[i-1] + np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
h = np.diff(t)  # Diferencias entre valores de t

# Paso 2: Construir el sistema de ecuaciones para M (segunda derivada)
A = np.zeros((n-1, n-1))
bx_vec = np.zeros(n-1)
by_vec = np.zeros(n-1)

for i in range(1, n):
    A[i-1, i-1] = 2 * (h[i-1] + h[i])
    if i-1 > 0:
        A[i-1, i-2] = h[i-1]
    if i < n-1:
        A[i-1, i] = h[i]
    bx_vec[i-1] = (6/h[i]) * (x[i+1] - x[i]) - (6/h[i-1]) * (x[i] - x[i-1])
    by_vec[i-1] = (6/h[i]) * (y[i+1] - y[i]) - (6/h[i-1]) * (y[i] - y[i-1])

# Paso 3: Resolver el sistema para obtener M_x y M_y
Mx = np.zeros(n+1)
My = np.zeros(n+1)
Mx[1:n] = np.linalg.solve(A, bx_vec)
My[1:n] = np.linalg.solve(A, by_vec)

# Paso 4: Calcular coeficientes a, b, c, d para cada subintervalo
ax = (Mx[1:] - Mx[:-1]) / (6 * h)
bx_coef = Mx[:-1] / 2
cx = (x[1:] - x[:-1]) / h - (2 * Mx[:-1] + Mx[1:]) * h / 6
dx_coef = x[:-1]

ay = (My[1:] - My[:-1]) / (6 * h)
by_coef = My[:-1] / 2
cy = (y[1:] - y[:-1]) / h - (2 * My[:-1] + My[1:]) * h / 6
dy_coef = y[:-1]

# Paso 5: Evaluar la interpolación en puntos intermedios de t
t_fino = np.linspace(t[0], t[-1], 10000)
x_fino = np.zeros_like(t_fino)
y_fino = np.zeros_like(t_fino)

for i in range(n):
    mask = (t_fino >= t[i]) & (t_fino <= t[i+1])
    ti = t_fino[mask] - t[i]
    x_fino[mask] = ax[i] * ti**3 + bx_coef[i] * ti**2 + cx[i] * ti + dx_coef[i]
    y_fino[mask] = ay[i] * ti**3 + by_coef[i] * ti**2 + cy[i] * ti + dy_coef[i]

# Paso 6: Graficar la curva interpolada
plt.plot(x_fino, y_fino, label="Spline Cúbico Paramétrico", color="blue")
plt.scatter(x, y, color="red", label="Puntos con incertidumbre")

# Parámetro para los "caps" de las barras de error (longitud de la línea perpendicular)
cap_size = 0.05  # Ajusta este valor según convenga

# Dibujar las barras de error direccionales para cada punto, con caps al estilo de errorbar
for xi, yi, sig, theta in zip(x, y, sigma_dir, theta_dir):
    # Calcular la barra central
    dx_err = sig * np.cos(theta)
    dy_err = sig * np.sin(theta)
    
    # Puntos extremo inferior y superior de la barra
    x_lower = xi - dx_err
    y_lower = yi - dy_err
    x_upper = xi + dx_err
    y_upper = yi + dy_err
    
    # Dibujar la barra central (línea sólida)
    plt.plot([x_lower, x_upper], [y_lower, y_upper],
             color="black", linestyle="-", linewidth=1)
    
    # Calcular el vector perpendicular normalizado para los caps
    norm = np.sqrt(dx_err**2 + dy_err**2)
    cap_dx = cap_size * (-dy_err) / norm
    cap_dy = cap_size * (dx_err) / norm
    
    # Dibujar cap inferior
    plt.plot([x_lower - cap_dx, x_lower + cap_dx],
             [y_lower - cap_dy, y_lower + cap_dy],
             color="black", linestyle="-", linewidth=1)
    
    # Dibujar cap superior
    plt.plot([x_upper - cap_dx, x_upper + cap_dx],
             [y_upper - cap_dy, y_upper + cap_dy],
             color="black", linestyle="-", linewidth=1)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Interpolación con Spline Cúbico 2D con Incertidumbre direccional")
plt.legend()
plt.grid(True)
plt.show()