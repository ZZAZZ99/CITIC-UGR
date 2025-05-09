import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
import random

# ----------------------
# Función auxiliar: convertir coeficientes en string tipo "3t^2 + 2t + 1"
# ----------------------
def poly_to_latex(coeffs):
    terms = []
    deg = len(coeffs) - 1
    for i, a in enumerate(coeffs):
        if a == 0:
            continue
        power = deg - i
        if power == 0:
            term = f"{a}"
        elif power == 1:
            term = f"{a}t"
        else:
            term = f"{a}t^{power}"
        terms.append(term)
    return "$" + " + ".join(terms) + "$"

# ----------------------
# 1. Datos y parametrización
# ----------------------
n_points = 50
t = np.linspace(0, 1, n_points)

# ----------------------
# 2. Generar dos funciones aleatorias relacionadas
# ----------------------
def random_polynomial(t, degree_range=(2, 10), coef_range=(1, 10)):
    d = random.randint(*degree_range)
    coeffs = [random.randint(*coef_range) for _ in range(d + 1)]
    f_t = sum(coeffs[k] * t**k for k in range(d + 1))
    return f_t, d, coeffs

# f1: función base
f1_t, d1, coeffs1 = random_polynomial(t)

# f2: función diferente pero desplazada verticalmente respecto a f1
f2_raw, d2, coeffs2 = random_polynomial(t)
offset = (np.mean(f1_t) + 10.0) - np.mean(f2_raw)
f2_t = f2_raw + offset

# Construcción de bases
p1 = np.vstack([t, f1_t]).T
p2 = np.vstack([t, f2_t]).T

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
    solver='clarabel'
)

print(f"Grado d1 = {d1}, Coeficientes f1(t):", coeffs1)
print(f"Grado d2 = {d2}, Coeficientes f2(t):", coeffs2)
print("λ óptimas:", np.round(lambda_opt, 6))

# ----------------------
# 9. Reconstruir y graficar
# ----------------------
p_fit = lambda_opt[0]*p1 + lambda_opt[1]*p2

label_p1 = "p1: " + poly_to_latex(coeffs1)
label_p2 = "p2: " + poly_to_latex(coeffs2)

plt.figure(figsize=(10,6))
plt.plot(p1[:,0], p1[:,1], 'r--', label=label_p1)
plt.plot(p2[:,0], p2[:,1], 'g--', label=label_p2)
plt.plot(p_fit[:,0], p_fit[:,1], 'b-', lw=2, label='Ajuste óptimo')
plt.scatter(p_fit[:,0], p_fit[:,1], c='blue', s=10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimización de curvatura con bases polinómicas aleatorias")
plt.legend(loc='upper left', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()