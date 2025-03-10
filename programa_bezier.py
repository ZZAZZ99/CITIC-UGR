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

def plot_interpolation(points, w, num_points=100):
    t_values = np.linspace(0, 1, num_points)
    interpolated_points = np.array([bezier_interpolation(points, t, w) for t in t_values])

    plt.figure(figsize=(7,5))
    plt.plot(interpolated_points[:, 0], interpolated_points[:, 1], 'b-', linewidth=2, label='Interpolación de Bézier')
    plt.plot(*zip(*points), 'ro', markersize=6, label='Puntos dados')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolación con Polinomios de Bézier en R^2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejemplo de uso
control_points = [(0, 0), (1, 3), (2, 2), (0, 1)]  # Puntos de control ajustados para mejor forma
weights = [1, 5, 5, 1]  # Pesos de los puntos de control
plot_interpolation(control_points, weights)
