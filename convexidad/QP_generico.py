import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
import random

# ----------------------
# 1. Datos y parametrización
# ----------------------
n_points = 50
t = np.linspace(0, 1, n_points)

# ----------------------
# 2. Grado aleatorio y coeficientes
# ----------------------
# Escoge grado d entre 2 y 10
d = random.randint(2, 10)

# Genera coeficientes a_0 ... a_d en 1..10
coeffs = [random.randint(1, 10) for _ in range(d + 1)]

# Constantes de desplazamiento
c1, c2 = 0.0, 10.0

# Construye f(t) = a_0 + a_1 t + ... + a_d t^d
f_t = sum(coeffs[k] * t**k for k in range(d + 1))

# Define las dos bases p1 y p2
p1 = np.vstack([t, f_t + c1]).T
p2 = np.vstack([t, f_t + c2]).T

# ----------------------
# 3. Matriz de segunda diferencia D
# ----------------------
n = n_points
D = np.zeros((n-2, n))
for i in range(n-2):
    D[i, i]   =  1
    D[i, i+1] = -2
    D[i, i+2] =  1

# ----------------------
# 4. Segundas derivadas discretas
# ----------------------
dd = []
for P in (p1, p2):
    dd_x = D @ P[:,0]
    dd_y = D @ P[:,1]
    dd.append((dd_x, dd_y))

# ----------------------
# 5. Matriz H de curvatura (2×2)
# ----------------------
H = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        xi, yi = dd[i]
        xj, yj = dd[j]
        H[i,j] = xi @ xj + yi @ yj

# ----------------------
# 6. Formar QP
# ----------------------
P_qp = 2 * H
q_qp = np.zeros(2)

# ----------------------
# 7. Restricciones
# ----------------------
lb = np.zeros(2)
ub = np.ones(2)
A_eq = np.ones((1,2))
b_eq = np.array([1.0])

# ----------------------
# 8. Resolver con el solver elegido
# ----------------------
lambda_opt = solve_qp(
    P_qp, q_qp,
    None, None,
    A_eq, b_eq,
    lb, ub,
    solver='clarabel'   # o 'highs', 'osqp', etc.
)

print(f"Grado d = {d}")
print("Coeficientes f(t):", coeffs)
print("λ óptimas:", lambda_opt)
lambda_opt = np.round(lambda_opt, 6)
print("λ óptimas:", lambda_opt)

# ----------------------
# 9. Reconstruir y graficar
# ----------------------
p_fit = lambda_opt[0]*p1 + lambda_opt[1]*p2

plt.figure(figsize=(8,5))
plt.plot(p1[:,0], p1[:,1], 'r--', label='Base p1')
plt.plot(p2[:,0], p2[:,1], 'g--', label='Base p2')
plt.plot(p_fit[:,0], p_fit[:,1], 'b-', lw=2, label='Ajuste óptimo')
plt.scatter(p_fit[:,0], p_fit[:,1], c='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimización de curvatura con bases aleatorias")
plt.legend()
plt.grid(True)
plt.show()
