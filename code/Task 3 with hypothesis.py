import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 20  # Axon length
dx = 0.2  # Spatial step size
dt = 0.01  # Time step size
Nx = int(L / dx)  # Number of spatial points
x = np.linspace(0, L, Nx)  # Spatial grid

# Constants for velocities and parameter A
V_plus_0 = 1  # Initial velocity in + direction
V_minus_0 = 1  # Initial velocity in - direction
A_values = np.linspace(0, 10, 5)  # Values for A from 0 to 10

# Function to compute velocity V+ and V-
def compute_velocities(n_plus, n_minus, A):
    V_plus = V_plus_0 * np.exp(-A * n_plus)
    V_minus = -V_minus_0 * np.exp(-A * n_minus)
    return V_plus, V_minus

# Initialize the concentration profiles for n+ and n-
n_plus = np.zeros(Nx)
n_minus = np.zeros(Nx)

# Set initial concentration profiles for n+ and n-
n_plus[:] = 0.1 + 0.5 * np.sin(np.pi * x / L)
n_minus[:] = 0.1 + 0.5 * np.cos(np.pi * x / L)

# Plot V+ and V- for different values of A
plt.figure(figsize=(10, 6))

for A in A_values:
    V_plus, V_minus = compute_velocities(n_plus, n_minus, A)
    plt.plot(x, V_plus, label=f'V+ for A={A}')
    plt.plot(x, V_minus, label=f'V- for A={A}')

plt.xlabel('Position (x)')
plt.ylabel('Velocity')
plt.title('Velocities V+ and V- for Different Values of A')
plt.legend()
plt.grid(True)
plt.savefig(r'C:\Users\haris\PycharmProjects\MOD600\Plots and others\Task_3g_velocities.png')
plt.show()