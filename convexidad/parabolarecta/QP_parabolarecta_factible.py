import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from scipy.linalg import block_diag

# 1) Parámetros y nodos
n = 50
t = np.linspace(0, 1, n)
offset_x = 1.0
offset_y = 2.0

m, c1 = 2.0, 0.0   # recta:       y = m t + c1
a, c2 = 1.0, 1.0   # parábola:    y = a t^2 + c2

# 2) Definición de bases
p1_line = np.vstack([    t,     m*t + c1]).T
p1_para = np.vstack([    t, a*t**2 + c2 + offset_y]).T
p2_para = np.vstack([t+offset_x, a*t**2 + c2 + offset_y]).T
p2_line = np.vstack([t+offset_x,     m*t + c1]).T

# 3) Matriz de segunda diferencia
D = np.zeros((n-2, n))
for i in range(n-2):
    D[i, i]   =  1
    D[i, i+1] = -2
    D[i, i+2] =  1

# 4) Construcción de H_full (n×n) y q_full (n,)
def build_H_q(pA, pB):
    yA = pA[:,1]; yB = pB[:,1]
    delta = yA - yB                  # (n,)
    G = D @ np.diag(delta)           # (n-2)×n
    H = 2 * (G.T @ G)                # (n×n)
    d2B = D @ yB                     # (n-2,)
    q = 2 * (G.T @ d2B)              # (n,)
    return H, q

H1, q1 = build_H_q(p1_line, p1_para)
H2, q2 = build_H_q(p2_para,  p2_line)

# 5) Igualdad dura: continuidad de valor en el cruce
y1_L = p1_line[-1,1]; y1_P = p1_para[-1,1]
y2_P = p2_para[ 0,1]; y2_L = p2_line[ 0,1]
δ1 = y1_L - y1_P
δ2 = y2_P - y2_L
A_eq = np.zeros((1, 2*n))
A_eq[0, n-1]   =  δ1    # λ1_last
A_eq[0,   0+n] = -δ2    # λ2_0
b_eq = np.array([y2_L - y1_P])

# 6) Penalización soft de C¹: (d1 - d2)^2
#    d1 = λ1_last*m + (1-λ1_last)*2a  => = λ1_last*(m-2a) + 2a
#    d2 = λ2_0*(2a*0) + (1-λ2_0)*m    => = m - λ2_0*m
#    error = d1-d2 = λ1_last*(m-2a) + 2a - m + λ2_0*m
v = np.zeros(2*n)
v[n-1]   = (m - 2*a)   # coef de λ1_last
v[n+0]   = m           # coef de λ2_0
c  = 2*a - m           # resto constante: 2a - m

# Peso de la penalización (ajústalo pequeño, e.g. 1e-2)
gamma = 1e-2

# 7) Montaje del QP global (P y q)
P_qp = block_diag(H1, H2) + gamma * np.outer(v, v)
q_qp = np.concatenate([q1, q2]) - gamma * c * v

# 8) Bounds 0 ≤ λ_i ≤ 1
lb = np.zeros(2*n)
ub = np.ones(2*n)

# 9) Resolver
λ_opt = solve_qp(P_qp, q_qp,
                 None, None,
                 A_eq, b_eq,
                 lb, ub,
                 solver='highs')
if λ_opt is None:
    raise RuntimeError("QP infactible (aunque con penalización C¹ debería ser siempre factible)")

λ_opt = np.round(λ_opt, 6)
λ1 = λ_opt[:n]
λ2 = λ_opt[n:]

print("λ tramo 1:", λ1)
print("λ tramo 2:", λ2)

# 10) Reconstruir y graficar
fit1 = λ1[:,None]*p1_line + (1-λ1)[:,None]*p1_para
fit2 = λ2[:,None]*p2_para + (1-λ2)[:,None]*p2_line
p_all = np.vstack([fit1, fit2])

plt.figure(figsize=(10,5))
plt.plot(p1_line[:,0], p1_line[:,1], 'r--', alpha=0.4, label='línea tramo 1')
plt.plot(p1_para[:,0], p1_para[:,1], 'g--', alpha=0.4, label='parábola 1')
plt.plot(p2_para[:,0], p2_para[:,1], 'g--', alpha=0.4, label='parábola 2')
plt.plot(p2_line[:,0], p2_line[:,1], 'r--', alpha=0.4, label='línea tramo 2')
plt.plot(p_all[:,0],   p_all[:,1],   'b-', lw=2, label='ajuste continuo C¹ (soft)')
plt.scatter(p_all[:,0], p_all[:,1], c='blue', s=10)
plt.xlabel("x"); plt.ylabel("y")
plt.title("Dos tramos con continuidad de valor dura y C¹ penalizada")
plt.legend(fontsize='small')
plt.grid(True)
plt.show()
