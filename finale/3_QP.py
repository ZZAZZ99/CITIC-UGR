import math
import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

# Función para solicitar datos manuales
def solicitar_datos():
    n = int(input("Número de puntos (sin cerrar curva): "))
    x_central = []
    y_central = []
    sigma_dir = []
    theta_dir = []
    for i in range(n):
        x = float(input(f"Punto {i+1} - x: "))
        y = float(input(f"Punto {i+1} - y: "))
        s = float(input(f"Punto {i+1} - sigma (incertidumbre radial): "))
        t = float(input(f"Punto {i+1} - theta (dirección en radianes): "))
        x_central.append(x)
        y_central.append(y)
        sigma_dir.append(s)
        theta_dir.append(t)
    return (np.array(x_central), np.array(y_central), np.array(sigma_dir), np.array(theta_dir))

# Pregunta al usuario si usar datos por defecto o manuales
uso_default = input("¿Desea usar los puntos del código con sus incertidumbres por defecto? (s/n): ")
if uso_default.lower() in ('s','si','y','yes'):
    x_central = np.array([0, 1, 2, 1, 0])
    y_central = np.array([0, 1, 2, 3, 2])
    sigma_dir = np.array([0.15, 0.10, 0.20, 0.15, 0.10])
    theta_dir = np.array([np.pi/6, np.pi/4, np.pi/3, np.pi/8, np.pi/10])
else:
    x_central, y_central, sigma_dir, theta_dir = solicitar_datos()

cerrar_curva = lambda arr: np.append(arr, arr[0])

x = cerrar_curva(x_central)
y = cerrar_curva(y_central)
dx_err = sigma_dir * np.cos(theta_dir)
dy_err = sigma_dir * np.sin(theta_dir)
x_plus = cerrar_curva(x_central + dx_err)
y_plus = cerrar_curva(y_central + dy_err)
x_minus = cerrar_curva(x_central - dx_err)
y_minus = cerrar_curva(y_central - dy_err)

n = len(x) - 1

p_vals = [1, 3, 2, math.inf, 2]
if len(p_vals) != n:
    p_input = input("Ingrese valores de p separados por comas (o presione Enter para usar la distancia euclidiana): ")
    if p_input.strip():
        p_vals = [float(v) if v.lower()!='inf' else math.inf for v in p_input.split(',')]
    else:
        p_vals = [2] * n

# Cálculo del spline periódico
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
        A[i, i] = 2*(h[im] + h[i])
        A[i, ip] = h[i]
        A[i, im] = h[im]
        bx[i] = (6/h[i])*(x[ip] - x[i]) - (6/h[im])*(x[i] - x[im])
        by[i] = (6/h[i])*(y[ip] - y[i]) - (6/h[im])*(y[i] - y[im])
    Mx = np.append(np.linalg.solve(A, bx), 0)
    My = np.append(np.linalg.solve(A, by), 0)
    Mx[-1] = Mx[0]
    My[-1] = My[0]
    ax = (Mx[1:] - Mx[:-1])/(6*h)
    bx_ = Mx[:-1]/2
    cx = (x[1:] - x[:-1])/h - (2*Mx[:-1] + Mx[1:])*h/6
    dx_ = x[:-1]
    ay = (My[1:] - My[:-1])/(6*h)
    by_ = My[:-1]/2
    cy = (y[1:] - y[:-1])/h - (2*My[:-1] + My[1:])*h/6
    dy_ = y[:-1]
    return t, ax, bx_, cx, dx_, ay, by_, cy, dy_, h

def evaluar_spline_interval(t, ax, bx_, cx, dx_, ay, by_, cy, dy_, h, t_fino):
    x_fino = np.empty_like(t_fino)
    y_fino = np.empty_like(t_fino)
    for i in range(len(h)):
        mask = (t_fino >= t[i]) & (t_fino <= t[i+1])
        ti = t_fino[mask] - t[i]
        x_fino[mask] = ax[i]*ti**3 + bx_[i]*ti**2 + cx[i]*ti + dx_[i]
        y_fino[mask] = ay[i]*ti**3 + by_[i]*ti**2 + cy[i]*ti + dy_[i]
    return x_fino, y_fino

splines = {}
for key, xx, yy in [('central', x, y), ('plus', x_plus, y_plus), ('minus', x_minus, y_minus)]:
    splines[key] = calcular_spline_periodico(xx, yy, p_vals)

tpts = 500
res_fine = {}
for key, (t_k, ax, bx_, cx, dx_, ay, by_, cy, dy_, h_k) in splines.items():
    t_fine = np.linspace(t_k[0], t_k[-1], tpts)
    x_f, y_f = evaluar_spline_interval(t_k, ax, bx_, cx, dx_, ay, by_, cy, dy_, h_k, t_fine)
    res_fine[key] = (t_fine, x_f, y_f)

def draw_error_bar(ax_plot, xi, yi, dxe, dye, cap=0.03):
    dxv, dyv = dxe, dye
    norm = math.hypot(dxv, dyv)
    px, py = -dyv/norm*cap, dxv/norm*cap
    ax_plot.plot([xi-dxv, xi+dxv], [yi-dyv, yi+dye], 'k-', lw=1)
    ax_plot.plot([xi-dxv-px, xi-dxv+px], [yi-dyv-py, yi-dyv+py], 'k-', lw=1)
    ax_plot.plot([xi+dxv-px, xi+dxv+px], [yi+dye-py, yi+dyv+py], 'k-', lw=1)

t_fig1, ax1 = plt.subplots()
_, x_c, y_c = res_fine['central']
ax1.plot(x_c, y_c, label='Spline Central')
ax1.scatter(x[:-1], y[:-1], color='red', label='Puntos centrales')
for xi, yi, dxe, dye in zip(x[:-1], y[:-1], dx_err, dy_err):
    draw_error_bar(ax1, xi, yi, dxe, dye)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Spline Cúbico Paramétrico Periódico (Central)')
ax1.legend()
ax1.grid(True)

t_fig2, ax2 = plt.subplots()
_, x_p, y_p = res_fine['plus']
_, x_m, y_m = res_fine['minus']
_, x_c2, y_c2 = res_fine['central']
ax2.plot(x_p, y_p, label='Spline +σ dirección')
ax2.plot(x_m, y_m, label='Spline -σ dirección')
ax2.plot(x_c2, y_c2, '--', color='gray', label='Spline central')
ax2.scatter(x[:-1], y[:-1], color='red', label='Puntos centrales')
for xi, yi, dxe, dye in zip(x[:-1], y[:-1], dx_err, dy_err):
    draw_error_bar(ax2, xi, yi, dxe, dye)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Curvas con Incertidumbre Direccional')
ax2.legend()
ax2.grid(True)

p1 = np.vstack((res_fine['plus'][1], res_fine['plus'][2])).T
p2 = np.vstack((res_fine['minus'][1], res_fine['minus'][2])).T
n_q = len(p1)
D = np.zeros((n_q-2, n_q))
for i in range(n_q-2):
    D[i, i] = 1
    D[i, i+1] = -2
    D[i, i+2] = 1

dd = []
for P in (p1, p2):
    dd_x = D @ P[:,0]
    dd_y = D @ P[:,1]
    dd.append((dd_x, dd_y))

H = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        xi, yi = dd[i]
        xj, yj = dd[j]
        H[i, j] = xi @ xj + yi @ yj

P_qp = 2 * H
q_qp = np.zeros(2)
A_eq = np.ones((1, 2))
b_eq = np.array([1.0])
lb = np.zeros(2)
ub = np.ones(2)

lambda_opt = solve_qp(P_qp, q_qp, None, None, A_eq, b_eq, lb, ub, solver='highs')
print("λ óptimas:", lambda_opt)

p_fit = lambda_opt[0]*p1 + lambda_opt[1]*p2

plt.figure()
plt.plot(p_fit[:,0], p_fit[:,1], color='purple', lw=2, label='Curva mínima curvatura')
plt.plot(res_fine['plus'][1], res_fine['plus'][2], '--', label='Spline +σ')
plt.plot(res_fine['minus'][1], res_fine['minus'][2], '--', label='Spline -σ')
for xi, yi, dxe, dye in zip(x[:-1], y[:-1], dx_err, dy_err):
    draw_error_bar(plt.gca(), xi, yi, dxe, dye)
plt.scatter(x[:-1], y[:-1], color='red', label='Puntos centrales')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Combinación de Mínima Curvatura con Incertidumbre')
plt.legend()
plt.grid(True)
plt.show()
