"""
Module for solving the Crank-Nicolson method in one dimensional for 
harmonic potential well 
Author: Agnibha Hanra
Date: March 2025
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0    # Mass of the particle
L = 2.0    # Length of the spatial grid (larger to accommodate harmonic potential)
N = 500   # Number of spatial points
dx = L / (N - 1)  # Spatial step
x = np.linspace(-L/2, L/2, N)  # Centered around x=0 for harmonic potential
dt = 0.001 # Time step
T = 1.0    # Total time for simulation
steps = int(T / dt)  # Number of time steps

# Harmonic potential parameters
omega = 10.0  # Angular frequency
V = 0.5 * m * omega**2 * x**2  # Harmonic potential

# Initial Gaussian wave packet
x0 = 0.0       # Initial position (center of harmonic potential)
sigma = 0.1    # Width of the wave packet
k0 = 10.0      # Initial wavenumber
psi = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)  # Normalize

# Crank-Nicholson Coefficients (including potential)
alpha = 1j * hbar * dt / (2 * m * dx ** 2)
diag = (1 + 2 * alpha) * np.ones(N) + 1j * dt / (2 * hbar) * V
off_diag = -alpha * np.ones(N - 1)
A = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
B = np.diag((1 - 2 * alpha) * np.ones(N) - 1j * dt / (2 * hbar) * V) + np.diag(alpha * np.ones(N - 1), 1) + np.diag(alpha * np.ones(N - 1), -1)

# Time Evolution
def time_step(psi):
    b = B @ psi
    psi_new = np.linalg.solve(A, b)
    return psi_new

# Animation Setup
fig, ax = plt.subplots()
ax.set_xlim(-L/2, L/2)
ax.set_ylim(0, 10)
line, = ax.plot([], [], lw=2, label="Probability Density")
potential_line, = ax.plot(x, V, 'r--', label="Harmonic Potential")
ax.set_title("Gaussian Wave Packet in Harmonic Potential")
ax.set_xlabel("Position")
ax.set_ylabel("Probability Density / Potential")
ax.legend()

# Animation Function
def animate(i):
    global psi
    psi = time_step(psi)
    line.set_data(x, np.abs(psi) ** 2)
    return line, potential_line

# Creating and saving animation
output_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), "results", "1D_results")
os.makedirs(output_dir, exist_ok=True)  

output_file = os.path.join(output_dir, "1D-harmonic-potential.mp4")  # Save path
ani = FuncAnimation(fig, animate, frames=steps, interval=20, blit=True)

# Save Animation using FFMpeg
writer = FFMpegWriter(fps=30, bitrate=1800)
ani.save(output_file, writer=writer)

#plt.show()

