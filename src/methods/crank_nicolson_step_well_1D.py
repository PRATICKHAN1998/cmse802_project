"""
Module for solving the Crank-Nicolson method in one dimension for a step potential well.

This script simulates the time evolution of a Gaussian wave packet 
inside a step potential well using the Crank-Nicolson method.

The simulation considers different step heights and computes the 
transmission coefficient for each case.

Author: Agnibha Hanra  
Date: March 2025  
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0  # Mass of the particle
L = 1.0  # Length of the spatial domain
N = 500  # Number of spatial points
dx = L / (N - 1)  # Spatial step size
x = np.linspace(0, L, N)  # Spatial grid
dt = 0.001  # Time step
T_total = 1.0  
steps = int(T_total / dt)  

# Step Potential Parameters
a = L / 2  # Position of the step
V0_values = [10, 50, 200, 1000]  # Step heights for different cases
colors = ['blue', 'green', 'orange', 'red'] 

# Initial Gaussian wave packet
x0 = L / 4  # Initial position of the wave packet
sigma = 0.05  # Width of the wave packet
k0 = 50.0  # Initial wavenumber (momentum)
psi = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x)  # Wave function
psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)  # Normalize wave function


def setup_matrices(V0):
    """
    Set up the Crank-Nicolson matrices for a given step potential height.

    Parameters:
    V0 (float): Step potential height.

    Returns:
    tuple: Matrices A and B used in the Crank-Nicolson method and the potential array V.
    """
    V = np.zeros(N)
    V[x >= a] = V0  # Applying step potential
    alpha = 1j * hbar * dt / (2 * m * dx ** 2)
    diag = (1 + 2 * alpha + 1j * dt * V / (2 * hbar)) * np.ones(N)
    off_diag = -alpha * np.ones(N - 1)
    A = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    B = np.diag((1 - 2 * alpha - 1j * dt * V / (2 * hbar)) * np.ones(N)) + np.diag(alpha * np.ones(N - 1), 1) + np.diag(alpha * np.ones(N - 1), -1)
    return A, B, V


def calculate_transmission_coefficient(psi):
    """
    Compute the transmission coefficient by integrating the wavefunction beyond the step.

    Parameters:
    psi (numpy array): Wave function at a given time step.

    Returns:
    float: Transmission coefficient.
    """
    return np.sum(np.abs(psi[x >= a]) ** 2) * dx


# Initializing wave functions and Crank-Nicolson matrices for all cases
psi_all = [psi.copy() for _ in V0_values]
A_all, B_all = [], []
for V0 in V0_values:
    A, B, V = setup_matrices(V0)
    A_all.append(A)
    B_all.append(B)

# Animation Setup
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(-2, 100)
lines = [ax.plot([], [], lw=2, color=color, label=f"$V_0 = {V0}$")[0] for V0, color in zip(V0_values, colors)]
step_line = ax.axvline(x=a, color='black', linestyle='--', linewidth=3, label="Step Position $x = a$")
transmission_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, verticalalignment='top')
ax.set_title("Wave Packet Dynamics for Different Step Heights $V_0$")
ax.set_xlabel("Position $x$")
ax.set_ylabel("Probability Density $|\psi|^2$")
ax.legend()
ax.grid()


def animate(i):
    """
    Update the wave packet animation at each time step.

    Parameters:
    i (int): Current frame number.

    Returns:
    list: Updated plot elements.
    """
    transmission_coefficients = []
    for j in range(len(V0_values)):
        b = B_all[j] @ psi_all[j]
        psi_all[j] = np.linalg.solve(A_all[j], b)  # Solving for new wave function
        lines[j].set_data(x, np.abs(psi_all[j]) ** 2)  # Updating probability density
        T = calculate_transmission_coefficient(psi_all[j])  # Computing transmission coefficient
        transmission_coefficients.append(f"$V_0 = {V0_values[j]}$: $T = {T:.3f}$")
    transmission_text.set_text("\n".join(transmission_coefficients))  
    return lines + [step_line, transmission_text]


# Creating and saving animation
output_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), "results", "1D_results")
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "1D-step-potential.mp4")
ani = FuncAnimation(fig, animate, frames=steps, interval=50, blit=True)
writer = FFMpegWriter(fps=15, bitrate=1800)
ani.save(output_file, writer=writer)

