import numpy as np
import matplotlib.pyplot as plt

def bezier_interpolation(points, t, w):
    points = np.array(points)
    n = len(points) - 1
    if n == 2:  # Bézier cuadrático
        result = (1 - t)**2 * points[0] * w[0] + 2 * (1 - t) * t * points[1] * w[1] + t**2 * points[2] * w[2]
    elif n == 3:  # Bézier cúbico
        result = (1 - t)**3 * points[0] * w[0] + 3 * (1 - t)**2 * t * points[1] * w[1] + 3 * (1 - t) * t**2 * points[2] * w[2] + t**3 * points[3] * w[3]
    else:
        raise ValueError("Solo se soportan polinomios de Bézier cuadráticos y cúbicos.")
    return result

def plot_interpolations(points1, w1, points2, w2, num_points=100):
    t_values = np.linspace(0, 1, num_points)
    interpolated_points1 = np.array([bezier_interpolation(points1, t, w1) for t in t_values])
    interpolated_points2 = np.array([bezier_interpolation(points2, t, w2) for t in t_values])
    
    plt.figure(figsize=(7,5))
    
    # Primera interpolación (por ejemplo, cúbica)
    plt.plot(interpolated_points1[:, 0], interpolated_points1[:, 1], 'b-', linewidth=2, label='Interpolación Bézier (conjunto 1)')
    plt.plot(*zip(*points1), 'ro', markersize=6, label='Puntos dados (conjunto 1)')
    
    # Segunda interpolación (por ejemplo, cuadrática)
    plt.plot(interpolated_points2[:, 0], interpolated_points2[:, 1], 'g-', linewidth=2, label='Interpolación Bézier (conjunto 2)')
    plt.plot(*zip(*points2), 'mo', markersize=6, label='Puntos dados (conjunto 2)')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolación con Polinomios de Bézier en R^2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejemplo de uso

# Conjunto 1: interpolación cúbica (4 puntos)
control_points1 = [(0, 0), (1, 3), (2, 2), (0, 1)]
control_points1 = np.array(control_points1)
weights1 = [1, 1, 1, 1]

option = input("Choose between velocity continuity or tangent continuity for smoothly joining Béziers (1 or 2): ")

if option == "1":
    
    control_points2 = [control_points1[3],
                   2 * control_points1[3] - control_points1[2],
                   (5, 6),
                   (3, 2)]

    weights2 = [1, 1, 1, 1]
    plot_interpolations(control_points1, weights1, control_points2, weights2)

elif option == "2":

    beta = input("Enter parameter beta for the extra degree of freedom: ")
    beta = float(beta)
    control_points2 = [control_points1[3],
                    control_points1[3] + beta*(control_points1[3] - control_points1[2]),
                   (5, 6),
                   (3, 2)]

    weights2 = [1, 1, 1, 1]
    plot_interpolations(control_points1, weights1, control_points2, weights2)

else:
    print("Invalid option. Please choose either 1 or 2.")
