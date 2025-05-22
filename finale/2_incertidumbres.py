import math
import numpy as np
import matplotlib.pyplot as plt

# Datos originales
x_central = np.array([0, 1, 2, 1, 0])
y_central = np.array([0, 1, 2, 3, 2])
sigma_dir = np.array([0.15, 0.10, 0.20, 0.15, 0.10])
theta_dir = np.array([np.pi/6, np.pi/4, np.pi/3, np.pi/8, np.pi/10])

# Añadir punto final para cerrar
def cerrar_curva(arr):
    return np.append(arr, arr[0])

x = cerrar_curva(x_central)
y = cerrar_curva(y_central)

# Calcular desplazamientos por incertidumbre
dx_err = sigma_dir * np.cos(theta_dir)
dy_err = sigma_dir * np.sin(theta_dir)

# Añadir incertidumbre positiva y negativa
x_plus = cerrar_curva(x_central + dx_err)
y_plus = cerrar_curva(y_central + dy_err)
x_minus = cerrar_curva(x_central - dx_err)
y_minus = cerrar_curva(y_central - dy_err)

n = len(x) - 1
p_vals = [1, 3, 2, math.inf, 2]

def calcular_spline_periodico(x, y, p_vals):
    t = np.zeros(len(x))
    for i in range(1, len(x)):
        p = p_vals[i-1]
        dx = abs(x[i] - x[i-1])
        dy = abs(y[i] - y[i-1])
        if p == 1:
            dist = dx + dy
        elif p == 2:
            dist = np.hypot(dx, dy)
        elif p == math.inf:
            dist = max(dx, dy)
        else:
            dist = (dx**p + dy**p)**(1/p)
        t[i] = t[i-1] + dist
    h = np.diff(t)

    A = np.zeros((n, n))
    bx = np.zeros(n)
    by = np.zeros(n)
    for i in range(n):
        ip = (i + 1) % n
        im = (i - 1) % n
        A[i, i] = 2 * (h[im] + h[i])
        A[i, ip] = h[i]
        A[i, im] = h[im]
        bx[i] = (6/h[i])*(x[ip] - x[i]) - (6/h[im])*(x[i] - x[im])
        by[i] = (6/h[i])*(y[ip] - y[i]) - (6/h[im])*(y[i] - y[im])

    Mx = np.append(np.linalg.solve(A, bx), 0)
    My = np.append(np.linalg.solve(A, by), 0)
    Mx[-1] = Mx[0]
    My[-1] = My[0]

    ax = (Mx[1:] - Mx[:-1]) / (6*h)
    bx_ = Mx[:-1] / 2
    cx = (x[1:] - x[:-1]) / h - (2*Mx[:-1] + Mx[1:]) * h / 6
    dx_ = x[:-1]

    ay = (My[1:] - My[:-1]) / (6*h)
    by_ = My[:-1] / 2
    cy = (y[1:] - y[:-1]) / h - (2*My[:-1] + My[1:]) * h / 6
    dy_ = y[:-1]

    return t, ax, bx_, cx, dx_, ay, by_, cy, dy_, h

def evaluar_spline(t, ax, bx, cx, dx, ay, by, cy, dy, h):
    t_fino = np.linspace(t[0], t[-1], 10000)
    x_fino = np.empty_like(t_fino)
    y_fino = np.empty_like(t_fino)
    for i in range(len(h)):
        mask = (t_fino >= t[i]) & (t_fino <= t[i+1])
        ti = t_fino[mask] - t[i]
        x_fino[mask] = ax[i]*ti**3 + bx[i]*ti**2 + cx[i]*ti + dx[i]
        y_fino[mask] = ay[i]*ti**3 + by[i]*ti**2 + cy[i]*ti + dy[i]
    return x_fino, y_fino

def dibujar_barras_error(x, y, dx_err, dy_err, cap_size=0.05):
    for xi, yi, dxe, dye in zip(x, y, dx_err, dy_err):
        x1, x2 = xi - dxe, xi + dxe
        y1, y2 = yi - dye, yi + dye
        plt.plot([x1, x2], [y1, y2], color="black", linestyle="-", linewidth=1)
        norm = np.hypot(dxe, dye)
        cap_dx = cap_size * (-dye) / norm
        cap_dy = cap_size * (dxe) / norm
        plt.plot([x1 - cap_dx, x1 + cap_dx], [y1 - cap_dy, y1 + cap_dy], color="black", linewidth=1)
        plt.plot([x2 - cap_dx, x2 + cap_dx], [y2 - cap_dy, y2 + cap_dy], color="black", linewidth=1)

# Curva central (figura original)
t, ax, bx, cx, dx_, ay, by, cy, dy_, h = calcular_spline_periodico(x, y, p_vals)
x_fino, y_fino = evaluar_spline(t, ax, bx, cx, dx_, ay, by, cy, dy_, h)

plt.figure()
plt.plot(x_fino, y_fino, label="Spline Cíclico Paramétrico")
plt.scatter(x, y, color="red", label="Puntos (cerrados)")
dibujar_barras_error(x[:-1], y[:-1], dx_err, dy_err)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Spline Cúbico Paramétrico Periódico (Curva Central)")
plt.legend()
plt.grid(True)

# Figura con curvas desplazadas
t_p, ax_p, bx_p, cx_p, dx_p, ay_p, by_p, cy_p, dy_p, h_p = calcular_spline_periodico(x_plus, y_plus, p_vals)
t_m, ax_m, bx_m, cx_m, dx_m, ay_m, by_m, cy_m, dy_m, h_m = calcular_spline_periodico(x_minus, y_minus, p_vals)

x_fino_p, y_fino_p = evaluar_spline(t_p, ax_p, bx_p, cx_p, dx_p, ay_p, by_p, cy_p, dy_p, h_p)
x_fino_m, y_fino_m = evaluar_spline(t_m, ax_m, bx_m, cx_m, dx_m, ay_m, by_m, cy_m, dy_m, h_m)

# Segunda figura con incertidumbre y curva central punteada
plt.figure()
plt.plot(x_fino_p, y_fino_p, color="blue", label="Spline +σ dirección")
plt.plot(x_fino_m, y_fino_m, color="orange", label="Spline -σ dirección")
plt.plot(x_fino, y_fino, linestyle="--", color="C0", label="Spline central (referencia)")
plt.scatter(x, y, color="red", label="Puntos centrales")
dibujar_barras_error(x[:-1], y[:-1], dx_err, dy_err)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Curvas con Incertidumbre Direccional")
plt.legend()
plt.grid(True)


plt.show()
