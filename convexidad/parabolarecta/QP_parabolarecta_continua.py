import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from scipy.linalg import block_diag

# 1) Parámetros y nodos
n_points = 50
t = np.linspace(0, 1, n_points)
offset_x = 1.0
offset_y = 2.0

# 2) Defino curvas
m, c1 = 2.0, 0.0
a, c2 = 1.0, 1.0
p1_line  = np.vstack([t,     m*t + c1]).T
p1_para  = np.vstack([t,     a*t**2 + c2 + offset_y]).T
p2_para  = np.vstack([t+offset_x, a*t**2 + c2 + offset_y]).T
p2_line  = np.vstack([t+offset_x, m*t + c1]).T

# 3) Matriz de segunda diferencia D
n = n_points
D = np.zeros((n-2, n))
for i in range(n-2):
    D[i, i]   =  1
    D[i, i+1] = -2
    D[i, i+2] =  1

# 4) H1 y H2
def compute_H(pA, pB):
    dd = []
    for P in (pA, pB):
        dd_x = D @ P[:,0]
        dd_y = D @ P[:,1]
        dd.append((dd_x, dd_y))
    H = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            xi, yi = dd[i]
            xj, yj = dd[j]
            H[i,j] = xi @ xj + yi @ yj
    return H

H1 = compute_H(p1_line, p1_para)
H2 = compute_H(p2_para,  p2_line)

# 5) QP global
P_qp = 2 * block_diag(H1, H2)
q_qp = np.zeros(4)

# 6) Restricciones de igualdad
y1_end   = np.array([p1_line[-1,1], p1_para[-1,1]])
y2_start = np.array([p2_para[0,1],  p2_line[0,1]])

A_eq = np.array([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [y1_end[0], y1_end[1], -y2_start[0], -y2_start[1]]
])
b_eq = np.array([1.0, 1.0, 0.0])

# 7) Bounds
lb = np.zeros(4)
ub = np.ones(4)

# 8) Resolver QP
λ_opt = solve_qp(P_qp, q_qp,
                 None, None,
                 A_eq, b_eq,
                 lb, ub,
                 solver='highs')
λ_opt = np.round(λ_opt, 6)
print("λ óptimas combinadas:", λ_opt)

# 9) Reconstruir y graficar
λ1_0, λ1_1, λ2_0, λ2_1 = λ_opt
fit1 = λ1_0 * p1_line + λ1_1 * p1_para
fit2 = λ2_0 * p2_para + λ2_1 * p2_line
p_all = np.vstack([fit1, fit2])

# 10) Verificar derivabilidad
# derivada hacia la derecha (fit2[1] - fit2[0])
# derivada hacia la izquierda (fit1[-1] - fit1[-2])
v_izq = fit1[-1] - fit1[-2]
v_der = fit2[1] - fit2[0]

# Calculamos el error relativo de la derivada
error = np.linalg.norm(v_izq - v_der) / np.linalg.norm(v_izq)
tol = 1e-3

if error < tol:
    print("La curva es derivable en el punto de unión.")
else:
    print("La curva NO es derivable en el punto de unión.")
    print(f"Vector derivada izquierda: {v_izq}")
    print(f"Vector derivada derecha:  {v_der}")
    print(f"Error relativo: {error:.2e}")

# 11) Gráfica
plt.figure(figsize=(9,5))
plt.plot(p1_line[:,0], p1_line[:,1], 'r--', alpha=0.5, label='recta tramo 1')
plt.plot(p1_para[:,0], p1_para[:,1], 'g--', alpha=0.5, label='párab. tramo 1')
plt.plot(p2_para[:,0], p2_para[:,1], 'g--', alpha=0.5, label='párab. tramo 2')
plt.plot(p2_line[:,0], p2_line[:,1], 'r--', alpha=0.5, label='recta tramo 2')

plt.plot(p_all[:,0], p_all[:,1], 'b-', lw=2, label='curva continua óptima')
plt.scatter(p_all[:,0], p_all[:,1], c='blue', s=10)

plt.xlabel("x"); plt.ylabel("y")
plt.title("Resultado QP de unión continua de dos tramos convexos")
plt.legend(fontsize='small')
plt.grid(True)
plt.show()
