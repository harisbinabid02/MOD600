#import libraries
import numpy as np
import matplotlib.pyplot as plt
import sys

class AdvectionDiffusionSolver:
    def __init__(self, task, T, M, Ldom, Vp, Vm, D_0, N_0, N_L, sigma_0, sigma_L, k_p, k_m, k_, NTime):
        self.task = task
        self.T = T
        self.M = M
        self.Ldom = Ldom
        self.Vp = Vp
        self.Vm = Vm
        self.D_0 = D_0
        self.N_0 = N_0
        self.N_L = N_L
        self.sigma_0 = sigma_0
        self.sigma_L = sigma_L
        self.k_p = k_p
        self.k_m = k_m
        self.k_ = k_
        self.NTime = NTime
        self.H = 0.1

        # Delta x
        self.dx = Ldom / M

        # Define cell centers
        self.x = np.arange(self.dx * 0.5, Ldom + 0.5 * self.dx, self.dx)

        # Define vectors for initial data n, n_p, n_m
        self.n0 = self._fun_initial_n(self.x, N_0, N_L, Ldom)
        self.np_0 = np.zeros_like(self.n0)
        self.nm_0 = np.zeros_like(self.n0)

        # Initialize solution vectors
        self.n = np.zeros_like(self.n0)
        self.n_old = np.zeros_like(self.n0)
        self.np = np.zeros_like(self.n0)
        self.np_old = np.zeros_like(self.n0)
        self.nm = np.zeros_like(self.n0)
        self.nm_old = np.zeros_like(self.n0)

        # Define vectors for handling different cells
        self.J = np.arange(0, self.M)  # number the cells of the domain
        self.J1 = np.arange(1, self.M - 1)  # the interior cells
        self.J2 = np.arange(0, self.M - 1)  # numbering of the cell interfaces

        # Time step dt
        self.dt = self.T / self.NTime

    def _fun_initial_n(self, x, N_0, N_L, Ldom):
        return N_0 + (N_L - N_0) * (x / Ldom)

    def _check_CFLconstraint_adv(self, value):
        if value > 1:
            print("CFL constraint not satisfied for advection. Stopping the program. Increase number time steps.")
            sys.exit()

    def _check_CFLconstraint_diff(self, value):
        if value > 0.5:
            print("CFL constraint not satisfied for diffusion. Stopping the program. Increase number time steps.")
            sys.exit()

    def solve(self):
        # Useful quantities for the discrete scheme
        lambda_1 = (self.dt / self.dx)  # advective eq
        lambda_2 = (self.dt / self.dx ** 2)  # diffusive eq

        # CFL number should be <= 1 for stability
        CFL_number_adv = lambda_1 * max(abs(self.Vp), abs(self.Vm))
        CFL_number_diff = lambda_2 * self.D_0

        print('CFL_number_adv =', CFL_number_adv)
        self._check_CFLconstraint_adv(CFL_number_adv)
        print('CFL_number_diff =', CFL_number_diff)
        self._check_CFLconstraint_diff(CFL_number_diff)

        # Initialize state
        self.n_old = np.copy(self.n0)
        self.np_old = np.copy(self.np_0)
        self.nm_old = np.copy(self.nm_0)

        # Iterate through time steps
        for j in range(self.NTime):
            if self.task == 'C':
                self.H = 0.015

                # Calculate numerical fluxes at cell interfaces
                np_half = self.np_old[self.J2]
                nm_half = self.nm_old[self.J2 + 1]
                Dn_half = (1 / self.dx) * (self.n_old[self.J2 + 1] - self.n_old[self.J2])

                # Update solution using the numerical scheme
                self.n[self.J1] = (self.n_old[self.J1] +
                                   self.D_0 * lambda_1 * (Dn_half[self.J1] - Dn_half[self.J1 - 1]))
                self.np[self.J1] = (self.np_old[self.J1] -
                                    self.Vp * lambda_1 * (np_half[self.J1] - np_half[self.J1 - 1]))
                self.nm[self.J1] = (self.nm_old[self.J1] -
                                    self.Vm * lambda_1 * (nm_half[self.J1] - nm_half[self.J1 - 1]))

            elif self.task == 'D':
                # Calculate numerical fluxes at cell interfaces
                np_half = self.np_old[self.J2]
                nm_half = self.nm_old[self.J2 + 1]
                Dn_half = (1 / self.dx) * (self.n_old[self.J2 + 1] - self.n_old[self.J2])

                # Update solution using the numerical scheme
                self.n[self.J1] = (self.n_old[self.J1] +
                                   self.D_0 * lambda_1 * (Dn_half[self.J1] - Dn_half[self.J1 - 1]) -
                                   (1 + self.k_) * self.dt * self.n_old[self.J1])

                self.np[self.J1] = (self.np_old[self.J1] -
                                    self.Vp * lambda_1 * (np_half[self.J1] - np_half[self.J1 - 1]) +
                                    self.dt * self.n_old[self.J1])

                self.nm[self.J1] = (self.nm_old[self.J1] -
                                    self.Vm * lambda_1 * (nm_half[self.J1] - nm_half[self.J1 - 1]) +
                                    (self.k_) * self.dt * self.n_old[self.J1])

            elif self.task == 'E':
                self.H = 0.015
                # Calculate numerical fluxes at cell interfaces
                np_half = self.np_old[self.J2]
                nm_half = self.nm_old[self.J2 + 1]
                Dn_half = (1 / self.dx) * (self.n_old[self.J2 + 1] - self.n_old[self.J2])

                # Update solution using the numerical scheme
                self.n[self.J1] = (self.n_old[self.J1] +
                                   self.D_0 * lambda_1 * (Dn_half[self.J1] - Dn_half[self.J1 - 1]) +
                                   self.k_p * self.dt * self.np_old[self.J1 - 1] +
                                   self.k_m * self.dt * self.nm_old[self.J1 - 1])

                self.np[self.J1] = (self.np_old[self.J1] -
                                    self.Vp * lambda_1 * (np_half[self.J1] - np_half[self.J1 - 1]) -
                                    self.k_p * self.dt * self.np_old[self.J1 - 1])

                self.nm[self.J1] = (self.nm_old[self.J1] -
                                    self.Vm * lambda_1 * (nm_half[self.J1] - nm_half[self.J1 - 1]) -
                                    self.k_m * self.dt * self.nm_old[self.J1 - 1])

            elif self.task == 'F':
                # Calculate numerical fluxes at cell interfaces
                np_half = self.np_old[self.J2]
                nm_half = self.nm_old[self.J2 + 1]
                Dn_half = (1 / self.dx) * (self.n_old[self.J2 + 1] - self.n_old[self.J2])

                # Update solution using the numerical scheme
                self.n[self.J1] = (self.n_old[self.J1] +
                                   self.D_0 * lambda_1 * (Dn_half[self.J1] - Dn_half[self.J1 - 1]) -
                                   (1 + self.k_) * self.dt * self.n_old[self.J1] +
                                   self.k_p * self.dt * self.np_old[self.J1 - 1] +
                                   self.k_m * self.dt * self.nm_old[self.J1 - 1])

                self.np[self.J1] = (self.np_old[self.J1] -
                                    self.Vp * lambda_1 * (np_half[self.J1] - np_half[self.J1 - 1]) +
                                    self.dt * self.n_old[self.J1] -
                                    self.k_p * self.dt * self.np_old[self.J1 - 1])

                self.nm[self.J1] = (self.nm_old[self.J1] -
                                    self.Vm * lambda_1 * (nm_half[self.J1] - nm_half[self.J1 - 1]) +
                                    (self.k_) * self.dt * self.n_old[self.J1] -
                                    self.k_m * self.dt * self.nm_old[self.J1 - 1])


            elif self.task == 'G':
                self.A_ = 7
                # Define the numerical flux at the cell-interfaces
                np_half = 0.5 * (self.np_old[:-1] * np.exp(-self.A_ * self.np_old[:-1]) +
                                 self.np_old[1:] * np.exp(-self.A_ * self.np_old[1:]) +
                                 self.np_old[:-1] - self.np_old[1:])

                nm_half = 0.5 * (-self.nm_old[:-1] * np.exp(-self.A_ * self.nm_old[:-1]) -
                                 self.nm_old[1:] * np.exp(-self.A_ * self.nm_old[1:]) +
                                 self.nm_old[:-1] - self.nm_old[1:])

                Dn_half = (1 / self.dx) * (self.n_old[1:] - self.n_old[:-1])

                # Numerical scheme
                self.n[1:-1] = (self.n_old[1:-1] + self.D_0 * lambda_1 * (Dn_half[1:] - Dn_half[:-1])
                                - (1 + self.k_) * self.n_old[1:-1] * self.dt
                                + self.k_p * self.dt * self.np_old[:-2] + self.k_m * self.dt * self.nm_old[:-2])

                self.np[1:-1] = (
                            self.np_old[1:-1] - lambda_1 * (np_half[1:] - np_half[:-1]) + self.n_old[1:-1] * self.dt
                            - self.k_p * self.dt * self.np_old[:-2])

                self.nm[1:-1] = (self.nm_old[1:-1] - lambda_1 * (nm_half[1:] - nm_half[:-1]) + self.k_ * self.n_old[
                                                                                                         1:-1] * self.dt
                                 - self.k_m * self.dt * self.nm_old[:-2])

            self.n[0] = self.N_0
            self.np[0] = self.N_0 * self.sigma_0
            self.nm[0] = self.nm[1]

            self.n[-1] = self.N_L
            self.np[-1] = self.np[-2]
            self.nm[-1] = self.N_L * self.sigma_L

            # Update old values
            self.n_old = np.copy(self.n)
            self.np_old = np.copy(self.np)
            self.nm_old = np.copy(self.nm)

    def plot_solution(self):
        self.solve()
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        axs[0].plot(self.x, self.n0, '--r')
        axs[0].plot(self.x, self.np_0, '--g')
        axs[0].plot(self.x, self.nm_0, '--b')
        axs[0].plot(self.x, self.n + self.np + self.nm, '-c')
        axs[0].set_title("Density n, n_+, n_-")
        axs[0].set_ylabel("$n, n_+, n_-$")
        axs[0].set_xlabel("x")
        axs[0].axis([-0.1, self.Ldom + 0.1, -0.1, +0.25])
        axs[0].legend(['$n$', '$n_{+}$', '$n_{-}$', '$n+n_{+}+n_{-}$'])
        axs[0].grid(True)

        axs[1].plot(self.x, self.n0, '--r')
        axs[1].plot(self.x, self.n, '-r')
        axs[1].set_title("Diffusive transport")
        axs[1].set_ylabel("n")
        axs[1].set_xlabel("x")
        axs[1].axis([-0.1, self.Ldom + 0.1, -0.1, +0.25])
        axs[1].legend(['$n_0$', '$n$'])
        axs[1].grid(True)

        axs[2].plot(self.x, self.np_0, '--g')
        axs[2].plot(self.x, self.np, '-g')
        axs[2].set_title("Advect + direction")
        axs[2].set_ylabel("$n_+$")
        axs[2].set_xlabel("x")
        axs[2].axis([-0.1, self.Ldom + 0.1, -0.01, self.H])
        axs[2].legend(['$n_{+,0}$', '$n_{+}$'])
        axs[2].grid(True)

        axs[3].plot(self.x, self.nm_0, '--b')
        axs[3].plot(self.x, self.nm, '-b')
        axs[3].set_title("Advect - direction")
        axs[3].set_ylabel("$n_-$")
        axs[3].set_xlabel("x")
        axs[3].axis([-0.1, self.Ldom + 0.1, -0.01, self.H])
        axs[3].legend(['$n_{-,0}$', '$n_{-}$'])
        axs[3].grid(True)

        plt.show()

#C
solver = AdvectionDiffusionSolver(task = 'C',T=30, M=100, Ldom=20, Vp=1, Vm=-1, D_0=0.4, N_0=0.1, N_L=0.01,
                                  sigma_0=0.1, sigma_L=0.1, k_p=0.5, k_m=0.5, k_=1.0, NTime=60*5*2*2)
solver.plot_solution()

#D
solver = AdvectionDiffusionSolver(task = 'D',T=30, M=100, Ldom=20, Vp=1, Vm=-1, D_0=0.4, N_0=0.1, N_L=0.01,
                                  sigma_0=0.1, sigma_L=0.1, k_p=0.5, k_m=0.5, k_=1.0, NTime=60*5*2*2)
solver.plot_solution()

#E
solver = AdvectionDiffusionSolver(task = 'E',T=30, M=100, Ldom=20, Vp=1, Vm=-1, D_0=0.4, N_0=0.1, N_L=0.01,
                                  sigma_0=0.1, sigma_L=0.1, k_p=0.5, k_m=0.5, k_=1.0, NTime=60*5*2*2)
solver.plot_solution()

#F
solver = AdvectionDiffusionSolver(task = 'F',T=30, M=100, Ldom=20, Vp=1, Vm=-1, D_0=0.4, N_0=0.1, N_L=0.01,
                                  sigma_0=0.1, sigma_L=0.1, k_p=0.5, k_m=0.5, k_=1.0, NTime=60*5*2*2)
solver.solve()
solver.plot_solution()