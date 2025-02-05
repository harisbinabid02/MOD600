#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import sys

class AdvectionDiffusionSolver:
    def __init__(self, problem_type, total_time, grid_points, domain_length, advection_vel_pos, advection_vel_neg, diffusion_coeff, initial_conc, boundary_conc, sigma_start, sigma_end, rate_pos, rate_neg, rate_total, num_steps):
        self.problem_type = problem_type
        self.total_time = total_time
        self.grid_points = grid_points
        self.domain_length = domain_length
        self.advection_vel_pos = advection_vel_pos
        self.advection_vel_neg = advection_vel_neg
        self.diffusion_coeff = diffusion_coeff
        self.initial_conc = initial_conc
        self.boundary_conc = boundary_conc
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.rate_pos = rate_pos
        self.rate_neg = rate_neg
        self.rate_total = rate_total
        self.num_steps = num_steps
        self.step_size = 0.1

        # Grid spacing
        self.dx = domain_length / grid_points

        # Define grid points
        self.x = np.arange(self.dx * 0.5, domain_length + 0.5 * self.dx, self.dx)

        # Initial concentration profiles
        self.n0 = self._initialize_concentration(self.x, initial_conc, boundary_conc, domain_length)
        self.n_pos_0 = np.zeros_like(self.n0)
        self.n_neg_0 = np.zeros_like(self.n0)

        # Solution vectors for different concentration terms
        self.n = np.zeros_like(self.n0)
        self.n_previous = np.zeros_like(self.n0)
        self.n_pos = np.zeros_like(self.n0)
        self.n_pos_previous = np.zeros_like(self.n0)
        self.n_neg = np.zeros_like(self.n0)
        self.n_neg_previous = np.zeros_like(self.n0)

        # Cell numbering
        self.cells = np.arange(0, self.grid_points)
        self.interior_cells = np.arange(1, self.grid_points - 1)
        self.cell_interfaces = np.arange(0, self.grid_points - 1)

        # Time step
        self.dt = self.total_time / self.num_steps

    def _initialize_concentration(self, x, initial_conc, boundary_conc, domain_length):
        return initial_conc + (boundary_conc - initial_conc) * (x / domain_length)

    def _check_CFL_condition_advection(self, value):
        if value > 1:
            print("CFL condition for advection violated. Increase the number of time steps.")
            sys.exit()

    def _check_CFL_condition_diffusion(self, value):
        if value > 0.5:
            print("CFL condition for diffusion violated. Increase the number of time steps.")
            sys.exit()

    def solve(self):
        lambda_adv = (self.dt / self.dx)  # advective term
        lambda_diff = (self.dt / self.dx ** 2)  # diffusive term

        # CFL condition values
        CFL_adv = lambda_adv * max(abs(self.advection_vel_pos), abs(self.advection_vel_neg))
        CFL_diff = lambda_diff * self.diffusion_coeff

        print(f'CFL for advection: {CFL_adv}')
        self._check_CFL_condition_advection(CFL_adv)
        print(f'CFL for diffusion: {CFL_diff}')
        self._check_CFL_condition_diffusion(CFL_diff)

        # Initialize the state
        self.n_previous = np.copy(self.n0)
        self.n_pos_previous = np.copy(self.n_pos_0)
        self.n_neg_previous = np.copy(self.n_neg_0)

        # Loop over the time steps
        for step in range(self.num_steps):
            if self.problem_type == 'C':
                self.step_size = 0.015

                # Flux at cell interfaces
                np_half = self.n_pos_previous[self.cell_interfaces]
                nm_half = self.n_neg_previous[self.cell_interfaces + 1]
                Dn_half = (1 / self.dx) * (self.n_previous[self.cell_interfaces + 1] - self.n_previous[self.cell_interfaces])

                # Update the solution
                self.n[self.interior_cells] = (self.n_previous[self.interior_cells] +
                                               self.diffusion_coeff * lambda_adv * (Dn_half[self.interior_cells] - Dn_half[self.interior_cells - 1]))
                self.n_pos[self.interior_cells] = (self.n_pos_previous[self.interior_cells] -
                                                    self.advection_vel_pos * lambda_adv * (np_half[self.interior_cells] - np_half[self.interior_cells - 1]))
                self.n_neg[self.interior_cells] = (self.n_neg_previous[self.interior_cells] -
                                                    self.advection_vel_neg * lambda_adv * (nm_half[self.interior_cells] - nm_half[self.interior_cells - 1]))

            elif self.problem_type == 'D':
                # Flux at cell interfaces
                np_half = self.n_pos_previous[self.cell_interfaces]
                nm_half = self.n_neg_previous[self.cell_interfaces + 1]
                Dn_half = (1 / self.dx) * (self.n_previous[self.cell_interfaces + 1] - self.n_previous[self.cell_interfaces])

                # Update the solution
                self.n[self.interior_cells] = (self.n_previous[self.interior_cells] +
                                               self.diffusion_coeff * lambda_adv * (Dn_half[self.interior_cells] - Dn_half[self.interior_cells - 1]) -
                                               (1 + self.rate_total) * self.dt * self.n_previous[self.interior_cells])

                self.n_pos[self.interior_cells] = (self.n_pos_previous[self.interior_cells] -
                                                    self.advection_vel_pos * lambda_adv * (np_half[self.interior_cells] - np_half[self.interior_cells - 1]) +
                                                    self.dt * self.n_previous[self.interior_cells])

                self.n_neg[self.interior_cells] = (self.n_neg_previous[self.interior_cells] -
                                                    self.advection_vel_neg * lambda_adv * (nm_half[self.interior_cells] - nm_half[self.interior_cells - 1]) +
                                                    self.rate_total * self.dt * self.n_previous[self.interior_cells])

            elif self.problem_type == 'E':
                self.step_size = 0.015
                # Flux at cell interfaces
                np_half = self.n_pos_previous[self.cell_interfaces]
                nm_half = self.n_neg_previous[self.cell_interfaces + 1]
                Dn_half = (1 / self.dx) * (self.n_previous[self.cell_interfaces + 1] - self.n_previous[self.cell_interfaces])

                # Update the solution
                self.n[self.interior_cells] = (self.n_previous[self.interior_cells] +
                                               self.diffusion_coeff * lambda_adv * (Dn_half[self.interior_cells] - Dn_half[self.interior_cells - 1]) +
                                               self.rate_pos * self.dt * self.n_pos_previous[self.interior_cells - 1] +
                                               self.rate_neg * self.dt * self.n_neg_previous[self.interior_cells - 1])

                self.n_pos[self.interior_cells] = (self.n_pos_previous[self.interior_cells] -
                                                    self.advection_vel_pos * lambda_adv * (np_half[self.interior_cells] - np_half[self.interior_cells - 1]) -
                                                    self.rate_pos * self.dt * self.n_pos_previous[self.interior_cells - 1])

                self.n_neg[self.interior_cells] = (self.n_neg_previous[self.interior_cells] -
                                                    self.advection_vel_neg * lambda_adv * (nm_half[self.interior_cells] - nm_half[self.interior_cells - 1]) -
                                                    self.rate_neg * self.dt * self.n_neg_previous[self.interior_cells - 1])

            elif self.problem_type == 'F':
                # Flux at cell interfaces
                np_half = self.n_pos_previous[self.cell_interfaces]
                nm_half = self.n_neg_previous[self.cell_interfaces + 1]
                Dn_half = (1 / self.dx) * (self.n_previous[self.cell_interfaces + 1] - self.n_previous[self.cell_interfaces])

                # Update the solution
                self.n[self.interior_cells] = (self.n_previous[self.interior_cells] +
                                               self.diffusion_coeff * lambda_adv * (Dn_half[self.interior_cells] - Dn_half[self.interior_cells - 1]) -
                                               (1 + self.rate_total) * self.dt * self.n_previous[self.interior_cells] +
                                               self.rate_pos * self.dt * self.n_pos_previous[self.interior_cells - 1] +
                                               self.rate_neg * self.dt * self.n_neg_previous[self.interior_cells - 1])

                self.n_pos[self.interior_cells] = (self.n_pos_previous[self.interior_cells] -
                                                    self.advection_vel_pos * lambda_adv * (np_half[self.interior_cells] - np_half[self.interior_cells - 1]) +
                                                    self.dt * self.n_previous[self.interior_cells] -
                                                    self.rate_pos * self.dt * self.n_pos_previous[self.interior_cells - 1])

                self.n_neg[self.interior_cells] = (self.n_neg_previous[self.interior_cells] -
                                                    self.advection_vel_neg * lambda_adv * (nm_half[self.interior_cells] - nm_half[self.interior_cells - 1]) +
                                                    self.rate_total * self.dt * self.n_previous[self.interior_cells] -
                                                    self.rate_neg * self.dt * self.n_neg_previous[self.interior_cells - 1])

            elif self.problem_type == 'G':
                self.A_ = 7
                # Define the numerical flux at the cell-interfaces
                np_half = 0.5 * (self.n_pos_previous[:-1] * np.exp(-self.A_ * self.n_pos_previous[:-1]) +
                                 self.n_pos_previous[1:] * np.exp(-self.A_ * self.n_pos_previous[1:]) +
                                 self.n_pos_previous[:-1] - self.n_pos_previous[1:])

                nm_half = 0.5 * (-self.n_neg_previous[:-1] * np.exp(-self.A_ * self.n_neg_previous[:-1]) -
                                 self.n_neg_previous[1:] * np.exp(-self.A_ * self.n_neg_previous[1:]) +
                                 self.n_neg_previous[:-1] - self.n_neg_previous[1:])

                Dn_half = (1 / self.dx) * (self.n_previous[1:] - self.n_previous[:-1])

                # Numerical scheme
                self.n[1:-1] = (self.n_previous[1:-1] + self.diffusion_coeff * lambda_adv * (Dn_half[1:] - Dn_half[:-1])
                                - (1 + self.rate_total) * self.n_previous[1:-1] * self.dt
                                + self.rate_pos * self.dt * self.n_pos_previous[:-2] + self.rate_neg * self.dt * self.n_neg_previous[:-2])

                self.n_pos[1:-1] = (self.n_pos_previous[1:-1] - lambda_adv * (np_half[1:] - np_half[:-1]) + self.n_previous[1:-1] * self.dt
                                    - self.rate_pos * self.dt * self.n_pos_previous[:-2])

                self.n_neg[1:-1] = (self.n_neg_previous[1:-1] - lambda_adv * (nm_half[1:] - nm_half[:-1]) + self.rate_total * self.n_previous[1:-1] * self.dt
                                    - self.rate_neg * self.dt * self.n_neg_previous[:-2])

            # Boundary conditions
            self.n[0] = self.initial_conc
            self.n_pos[0] = self.initial_conc * self.sigma_start
            self.n_neg[0] = self.n_neg[1]

            self.n[-1] = self.boundary_conc
            self.n_pos[-1] = self.n_pos[-2]
            self.n_neg[-1] = self.boundary_conc * self.sigma_end

            # Update the previous values for the next iteration
            self.n_previous = np.copy(self.n)
            self.n_pos_previous = np.copy(self.n_pos)
            self.n_neg_previous = np.copy(self.n_neg)

    def plot_solution(self):
        self.solve()
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        axs[0].plot(self.x, self.n0, '--r')
        axs[0].plot(self.x, self.n_pos_0, '--g')
        axs[0].plot(self.x, self.n_neg_0, '--b')
        axs[0].plot(self.x, self.n + self.n_pos + self.n_neg, '-c')
        axs[0].set_title("Concentration profiles")
        axs[0].set_ylabel("Concentration")
        axs[0].set_xlabel("Position (x)")
        axs[0].axis([-0.1, self.domain_length + 0.1, -0.1, +0.25])
        axs[0].legend(['$n$', '$n_{+}$', '$n_{-}$', 'Total'])
        axs[0].grid(True)

        axs[1].plot(self.x, self.n0, '--r')
        axs[1].plot(self.x, self.n, '-r')
        axs[1].set_title("Diffusive Transport")
        axs[1].set_ylabel("n")
        axs[1].set_xlabel("x")
        axs[1].axis([-0.1, self.domain_length + 0.1, -0.1, +0.25])
        axs[1].legend(['$n_0$', '$n$'])
        axs[1].grid(True)

        axs[2].plot(self.x, self.n_pos_0, '--g')
        axs[2].plot(self.x, self.n_pos, '-g')
        axs[2].set_title("Advection + Direction")
        axs[2].set_ylabel("$n_{+}$")
        axs[2].set_xlabel("x")
        axs[2].axis([-0.1, self.domain_length + 0.1, -0.01, self.step_size])
        axs[2].legend(['$n_{+,0}$', '$n_{+}$'])
        axs[2].grid(True)

        axs[3].plot(self.x, self.n_neg_0, '--b')
        axs[3].plot(self.x, self.n_neg, '-b')
        axs[3].set_title("Advection - Direction")
        axs[3].set_ylabel("$n_{-}$")
        axs[3].set_xlabel("x")
        axs[3].axis([-0.1, self.domain_length + 0.1, -0.01, self.step_size])
        axs[3].legend(['$n_{-,0}$', '$n_{-}$'])
        axs[3].grid(True)
        # Create a unique filename based on the problem type
        filename = f'C:\\Users\\haris\\PycharmProjects\\MOD600\\Plots and others\\Task_3_Problem_{self.problem_type}.png'

        # Save the plot to the unique filename
        plt.savefig(filename)

        plt.show()

#Example for Task C
solver = AdvectionDiffusionSolver(problem_type='C', total_time=30, grid_points=100, domain_length=20, advection_vel_pos=1, advection_vel_neg=-1,
                                  diffusion_coeff=0.4, initial_conc=0.1, boundary_conc=0.01, sigma_start=0.1, sigma_end=0.1, rate_pos=0.5,
                                  rate_neg=0.5, rate_total=1.0, num_steps=60*5*2*2)
solver.plot_solution()

#Example for Task D
solver = AdvectionDiffusionSolver(problem_type='D', total_time=30, grid_points=100, domain_length=20, advection_vel_pos=1, advection_vel_neg=-1,
                                  diffusion_coeff=0.4, initial_conc=0.1, boundary_conc=0.01, sigma_start=0.1, sigma_end=0.1, rate_pos=0.5,
                                  rate_neg=0.5, rate_total=1.0, num_steps=60*5*2*2)
solver.plot_solution()

#Example for Task E
solver = AdvectionDiffusionSolver(problem_type='E', total_time=30, grid_points=100, domain_length=20, advection_vel_pos=1, advection_vel_neg=-1,
                                  diffusion_coeff=0.4, initial_conc=0.1, boundary_conc=0.01, sigma_start=0.1, sigma_end=0.1, rate_pos=0.5,
                                  rate_neg=0.5, rate_total=1.0, num_steps=60*5*2*2)
solver.plot_solution()

#Example for Task F
solver = AdvectionDiffusionSolver(problem_type='F', total_time=30, grid_points=100, domain_length=20, advection_vel_pos=1, advection_vel_neg=-1,
                                  diffusion_coeff=0.4, initial_conc=0.1, boundary_conc=0.01, sigma_start=0.1, sigma_end=0.1, rate_pos=0.5,
                                  rate_neg=0.5, rate_total=1.0, num_steps=60*5*2*2)
solver.plot_solution()

#Example for Task G
solver = AdvectionDiffusionSolver(problem_type='G', total_time=30, grid_points=100, domain_length=20, advection_vel_pos=1, advection_vel_neg=-1,
                                  diffusion_coeff=0.4, initial_conc=0.1, boundary_conc=0.01, sigma_start=0.1, sigma_end=0.1, rate_pos=0.5,
                                  rate_neg=0.5, rate_total=1.0, num_steps=60*5*2*2)
solver.plot_solution()