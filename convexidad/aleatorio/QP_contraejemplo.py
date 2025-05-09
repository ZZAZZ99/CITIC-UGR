import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

# ----------------------
# Definición explícita de los dos polinomios
# ----------------------
n_points = 100
t = np.linspace(0, 1, n_points)

# p1(t) = 10 t^2
coeffs1 = [0.0, 0.0, 10.0]  # grado 2
f1_t = coeffs1[2] * t**2

# p2(t) = 0.1 t^5
coeffs2 = [0.0]*5 + [0.1]   # grado 5
f2_t = coeffs2[5] * t**5

# Bases en el plano
p1 = np.vstack([t, f1_t]).T
p2 = np.vstack([t, f2_t]).T

# ----------------------
# Cálculo de J1 y J2 para verificar el contraejemplo
# ----------------------
# Matriz de segunda diferencia D
n = n_points
D = np.zeros((n-2, n))
for i in range(n-2):
    D[i, i]   =  1
    D[i, i+1] = -2
    D[i, i+2] =  1

# Segundas derivadas discretas
dd1 = D @ p1[:,1]
dd2 = D @ p2[:,1]

# Curvaturas efectivas
J1 = np.dot(dd1, dd1)
J2 = np.dot(dd2, dd2)

print(f"J1 (grado 2) = {J1:.4f}")
print(f"J2 (grado 5) = {J2:.4f}")
# Aquí veremos J1 >> J2

# ----------------------
# QP: combinación convexa que minimiza curvatura
# ----------------------
# Construcción de H
H = np.array([
    [np.dot(dd1, dd1), np.dot(dd1, dd2)],
    [np.dot(dd2, dd1), np.dot(dd2, dd2)]
])

P_qp = 2 * H
q_qp = np.zeros(2)
A_eq = np.ones((1,2))
b_eq = np.array([1.0])
lb = np.zeros(2)
ub = np.ones(2)

lambda_opt = solve_qp(
    P_qp, q_qp,
    None, None,
    A_eq, b_eq,
    lb, ub,
    solver='highs'
)
lambda_opt = np.round(lambda_opt, 6)
print("λ óptimas:", lambda_opt)

# ----------------------
# Graficar
# ----------------------
p_fit = lambda_opt[0]*p1 + lambda_opt[1]*p2

plt.figure(figsize=(8,5))
plt.plot(p1[:,0], p1[:,1], 'r--', label="p1(t)=10t²")
plt.plot(p2[:,0], p2[:,1], 'g--', label="p2(t)=0.1t⁵")
plt.plot(p_fit[:,0], p_fit[:,1], 'b-', lw=2, label='Ajuste QP')
plt.scatter(p_fit[:,0], p_fit[:,1], c='blue', s=10)
plt.xlabel("t")
plt.ylabel("y")
plt.title("Contraejemplo: grado 2 vs. grado 5")
plt.legend()
plt.grid(True)
plt.show()
