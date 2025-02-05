#Task 2 First solving exact solutions with MOC method
#And than FDM methods for numerical methods

import numpy as np
import matplotlib.pyplot as plt

# Define parameters
a = 1  # for problem I, we assume a = 1 (this can be adjusted)
T = 0.5  # final time for all problems
x_min, x_max = 0, 2  # spatial domain
M = 500  # number of grid points
dx = (x_max - x_min) / M  # grid spacing
x = np.linspace(x_min, x_max, M + 1)  # grid points
dt = 0.1 * dx / a  # time step (based on stability condition)


# Initial condition u0(x)
def u0(x):
    return np.where((x >= 0.8) & (x <= 1.2), 1, 0)


# Problem I: u_t + a u_x = 0 (MOC and numerical solution)
def exact_solution_I(x, t):
    return u0(x - a * t)


def numerical_solution_I(M, T, a, dt, dx):
    # Number of time steps
    N = int(T / dt)
    u = np.zeros((N + 1, M + 1))
    u[0, :] = u0(x)  # initial condition

    for n in range(0, N):
        for j in range(1, M):
            u[n + 1, j] = u[n, j] - a * dt / dx * (u[n, j] - u[n, j - 1])

    return u


# Problem II: u_t + x u_x = 0 (MOC and numerical solution)
def exact_solution_II(x, t):
    return u0(x * np.exp(-t))


def numerical_solution_II(M, T, dt, dx):
    N = int(T / dt)
    u = np.zeros((N + 1, M + 1))
    u[0, :] = u0(x)

    for n in range(0, N):
        for j in range(1, M):
            u[n + 1, j] = u[n, j] - x[j] * dt / dx * (u[n, j] - u[n, j - 1])

    return u


# Problem III: u_t + u_x = x (MOC and numerical solution)
def exact_solution_III(x, t):
    return u0(x - t) + x * t - 0.5 * t ** 2


def numerical_solution_III(M, T, dt, dx):
    N = int(T / dt)
    u = np.zeros((N + 1, M + 1))
    u[0, :] = u0(x)

    for n in range(0, N):
        for j in range(1, M):
            u[n + 1, j] = u[n, j] - dt / dx * (u[n, j] - u[n, j - 1]) + x[j] * dt

    return u


# Solve numerically for all three problems
u_numerical_I = numerical_solution_I(M, T, a, dt, dx)
u_numerical_II = numerical_solution_II(M, T, dt, dx)
u_numerical_III = numerical_solution_III(M, T, dt, dx)

# Exact solutions for comparison
u_exact_I = exact_solution_I(x, T)
u_exact_II = exact_solution_II(x, T)
u_exact_III = exact_solution_III(x, T)

# Plot results for Problem I
plt.figure(figsize=(10, 6))
plt.plot(x, u_exact_I, label="Exact Solution I", color='blue')
plt.plot(x, u_numerical_I[-1, :], label="Numerical Solution I", linestyle='dashed', color='red')
plt.title("Problem I: Exact vs Numerical Solution")
plt.xlabel("x")
plt.ylabel("u(x, T)")
plt.legend()
plt.grid()
plt.savefig(r'C:\Users\haris\PycharmProjects\MOD600\Plots and others\Task_2_Problem1.png')
plt.show()

# Plot results for Problem II
plt.figure(figsize=(10, 6))
plt.plot(x, u_exact_II, label="Exact Solution II", color='blue')
plt.plot(x, u_numerical_II[-1, :], label="Numerical Solution II", linestyle='dashed', color='red')
plt.title("Problem II: Exact vs Numerical Solution")
plt.xlabel("x")
plt.ylabel("u(x, T)")
plt.legend()
plt.grid()
plt.savefig(r'C:\Users\haris\PycharmProjects\MOD600\Plots and others\Task_2_Problem2.png')
plt.show()

# Plot results for Problem III
plt.figure(figsize=(10, 6))
plt.plot(x, u_exact_III, label="Exact Solution III", color='blue')
plt.plot(x, u_numerical_III[-1, :], label="Numerical Solution III", linestyle='dashed', color='red')
plt.title("Problem III: Exact vs Numerical Solution")
plt.xlabel("x")
plt.ylabel("u(x, T)")
plt.legend()
plt.grid()
plt.savefig(r'C:\Users\haris\PycharmProjects\MOD600\Plots and others\Task_2_Problem3.png')
plt.show()