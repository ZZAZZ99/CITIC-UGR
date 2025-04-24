import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

# ----------------------
# 1) Datos
# ----------------------
n_points = 50
t = np.linspace(0, 1, n_points)

# ----------------------
# 2) Bases: dos parábolas y una línea horizontal
# ----------------------
a, b = 1.0, 2.0     # p1, p2: -a t^2 + b t + c
c1, c2 = 0.0, 1.0
p1 = np.vstack([t, -a*t**2 + b*t + c1]).T
p2 = np.vstack([t, -a*t**2 + b*t + c2]).T
p3 = np.vstack([t, np.ones_like(t)]).T      # línea de curvatura cero

# ----------------------
# 3) Matriz de segunda diferencia D
# ----------------------
n = n_points
D = np.zeros((n-2, n))
for i in range(n-2):
    D[i, i]   = 1
    D[i, i+1] = -2
    D[i, i+2] = 1

# ----------------------
# 4) Secundas derivadas discretas de cada base
# ----------------------
dd = []
for P in (p1, p2, p3):
    dd_x = D @ P[:,0]
    dd_y = D @ P[:,1]
    dd.append((dd_x, dd_y))

# ----------------------
# 5) Matriz H de curvatura
#    H[i,j] = sum_k (dd_i_x[k]*dd_j_x[k] + dd_i_y[k]*dd_j_y[k])
# ----------------------
H = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        xi, yi = dd[i]
        xj, yj = dd[j]
        H[i,j] = xi @ xj + yi @ yj

# ----------------------
# 6) Formar QP: P = 2H, q = [0,0,-α]
# ----------------------
P_qp = 2 * H
# Añadimos un término lineal que premie λ3:
alpha = 1.0
q_qp = np.array([0.0, 0.0, -alpha])

# ----------------------
# 7) Restricciones
# ----------------------
lb = np.zeros(3)
ub = np.ones(3)
# Suma de lambdas = 1 (convexidad global)
A_eq = np.ones((1,3))
b_eq = np.array([1.0])

# ----------------------
# 8) Resolver
# ----------------------
lambda_opt = solve_qp(
    P_qp, q_qp,
    None, None,
    A_eq, b_eq,
    lb, ub,
    solver='osqp'
)

lambda_opt = np.where(np.abs(lambda_opt) < 1e-3, 0, lambda_opt)
lambda_opt = np.where(np.abs(lambda_opt) > 1, 1, lambda_opt)

print("λ óptimas:", lambda_opt)
# → debería salir [0, 0, 1]

# ----------------------
# 9) Reconstruir y graficar
# ----------------------
p_fit = lambda_opt[0]*p1 + lambda_opt[1]*p2 + lambda_opt[2]*p3

plt.figure(figsize=(8,5))
plt.plot(p1[:,0], p1[:,1], 'r--', label='Parábola 1')
plt.plot(p2[:,0], p2[:,1], 'g--', label='Parábola 2')
plt.plot(p3[:,0], p3[:,1], 'k--', label='Línea horizontal')
plt.plot(p_fit[:,0], p_fit[:,1], 'b-', lw=2, label='Óptimo (λ3=1)')
plt.scatter(p_fit[:,0], p_fit[:,1], c='blue')
plt.xlabel("x"); plt.ylabel("y")
plt.title("Minimización de curvatura con sesgo a la base horizontal")
plt.legend(); plt.grid(True); plt.show()
