import math
import numpy as np
import matplotlib.pyplot as plt

# Definir los puntos en el plano (x, y) y hacer la curva cerrada
x = np.array([0, 1, 2, 1, 0])  # Coordenadas x
y = np.array([0, 1, 2, 3, 2])  # Coordenadas y

# Añadir el punto inicial al final para cerrar la curva
x = np.append(x, x[0])
y = np.append(y, y[0])

n = len(x) - 1  # Ahora n subintervalos incluyendo el último que cierra

# Definir aquí los órdenes p para cada uno de los n intervalos
p_vals = [1, 3, 2, math.inf, 2]  # Debe tener longitud n

# Paso 1: Calcular el parámetro t usando distancia acumulada (norma p)
t = np.zeros(len(x))
for i in range(1, len(x)):
    p = p_vals[i-1]
    dx = abs(x[i] - x[i-1])
    dy = abs(y[i] - y[i-1])
    if p == 1:
        dist = dx + dy
    elif p == 2:
        dist = np.hypot(x[i] - x[i-1], y[i] - y[i-1])
    elif p == math.inf:
        dist = max(dx, dy)
    else:
        dist = (dx**p + dy**p)**(1/p)
    t[i] = t[i-1] + dist

h = np.diff(t)

# Paso 2 (PERIÓDICO): Montar sistema cíclico de tamaño n x n para M[0..n-1]
A = np.zeros((n, n))
bx = np.zeros(n)
by = np.zeros(n)
# Rellenar A y b
for i in range(n):
    ip = (i + 1) % n
    im = (i - 1) % n
    A[i, i] = 2 * (h[im] + h[i])
    A[i, ip] = h[i]
    A[i, im] = h[im]
    # Términos independientes
    bx[i] = (6/h[i])*(x[ip] - x[i]) - (6/h[im])*(x[i] - x[im])
    by[i] = (6/h[i])*(y[ip] - y[i]) - (6/h[im])*(y[i] - y[im])

# Paso 3: Resolver para M periódicas
M_cycle_x = np.linalg.solve(A, bx)
M_cycle_y = np.linalg.solve(A, by)

# Construir el vector M completo con M[n]=M[0]
Mx = np.append(M_cycle_x, M_cycle_x[0])
My = np.append(M_cycle_y, M_cycle_y[0])

# Paso 4: Coeficientes de cada subintervalo
ax = (Mx[1:] - Mx[:-1]) / (6*h)
bx = Mx[:-1] / 2
cx = (x[1:] - x[:-1]) / h - (2*Mx[:-1] + Mx[1:]) * h / 6
dx = x[:-1]

ay = (My[1:] - My[:-1]) / (6*h)
by = My[:-1] / 2
cy = (y[1:] - y[:-1]) / h - (2*My[:-1] + My[1:]) * h / 6
dy = y[:-1]

# Mostrar ecuaciones para cada intervalo
def print_spline(p_vals, t, ax, bx, cx, dx, ay, by, cy, dy):
    for i in range(n):
        print(f"Intervalo [{t[i]:.3f}, {t[i+1]:.3f}], p={p_vals[i]}")
        print(f"  x(t) = {ax[i]:.6f}(t-{t[i]:.3f})³ + {bx[i]:.6f}(t-{t[i]:.3f})² + {cx[i]:.6f}(t-{t[i]:.3f}) + {dx[i]:.6f}")
        print(f"  y(t) = {ay[i]:.6f}(t-{t[i]:.3f})³ + {by[i]:.6f}(t-{t[i]:.3f})² + {cy[i]:.6f}(t-{t[i]:.3f}) + {dy[i]:.6f}\n")
print_spline(p_vals, t, ax, bx, cx, dx, ay, by, cy, dy)

# Paso 5: Evaluar la curva cerrada
t_fino = np.linspace(t[0], t[-1], 10000)
x_fino = np.empty_like(t_fino)
y_fino = np.empty_like(t_fino)
for i in range(n):
    mask = (t_fino >= t[i]) & (t_fino <= t[i+1])
    ti = t_fino[mask] - t[i]
    x_fino[mask] = ax[i]*ti**3 + bx[i]*ti**2 + cx[i]*ti + dx[i]
    y_fino[mask] = ay[i]*ti**3 + by[i]*ti**2 + cy[i]*ti + dy[i]

# Paso 6: Graficar curva periódica
plt.figure()
plt.plot(x_fino, y_fino, label="Spline Cíclico Paramétrico")
plt.scatter(x, y, color="red", label="Puntos (cerrados)")

# Dibujar barras de error direccionales
sigma_dir = np.array([0.15, 0.10, 0.20, 0.15, 0.10])  # Longitud de la incertidumbre
theta_dir = np.array([np.pi/6, np.pi/4, np.pi/3, np.pi/8, np.pi/10])  # Ángulo en radianes

cap_size = 0.05  # Tamaño de las "capas" en los extremos de la barra

for xi, yi, sig, theta in zip(x[:-1], y[:-1], sigma_dir, theta_dir):
    dx_err = sig * np.cos(theta)
    dy_err = sig * np.sin(theta)

    x_lower = xi - dx_err
    y_lower = yi - dy_err
    x_upper = xi + dx_err
    y_upper = yi + dy_err

    # Línea principal
    plt.plot([x_lower, x_upper], [y_lower, y_upper], color="black", linestyle="-", linewidth=1)

    # Capas (pequeños segmentos perpendiculares en los extremos)
    norm = np.sqrt(dx_err**2 + dy_err**2)
    cap_dx = cap_size * (-dy_err) / norm
    cap_dy = cap_size * (dx_err) / norm

    plt.plot([x_lower - cap_dx, x_lower + cap_dx],
             [y_lower - cap_dy, y_lower + cap_dy], color="black", linewidth=1)
    plt.plot([x_upper - cap_dx, x_upper + cap_dx],
             [y_upper - cap_dy, y_upper + cap_dy], color="black", linewidth=1)


plt.xlabel("x")
plt.ylabel("y")
plt.title("Spline Cúbico Paramétrico Periódico (Curva Cerrada)")
plt.legend()
plt.grid(True)
plt.show()