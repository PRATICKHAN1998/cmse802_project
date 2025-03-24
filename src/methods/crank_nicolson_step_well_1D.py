"""
Module for solving the Crank-Nicolson method in one dimensional for 
step potential well
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
L = 1.0    # Length of the domain
N = 500   # Number of spatial points
dx = L / (N - 1)  # Spatial step
x = np.linspace(0, L, N)
dt = 0.001 # Time step
T_total = 1.0    # Total time for simulation
steps = int(T_total / dt)  # Number of time steps

# Step Potential Parameters
a = L / 2  # Position of the step
V0_values = [10, 50, 200, 1000]  # Four cases: small, moderate, large, very large
colors = ['blue', 'green', 'orange', 'red']  # Colors for each case

# Initial Gaussian wave packet
x0 = L / 4       # Initial position (left of the step)
sigma = 0.05     # Width of the wave packet
k0 = 50.0        # Initial wavenumber
psi = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)  # Normalize

# Function to set up Crank-Nicolson matrices
def setup_matrices(V0):
    V = np.zeros(N)
    V[x >= a] = V0
    alpha = 1j * hbar * dt / (2 * m * dx ** 2)
    diag = (1 + 2 * alpha + 1j * dt * V / (2 * hbar)) * np.ones(N)
    off_diag = -alpha * np.ones(N - 1)
    A = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    B = np.diag((1 - 2 * alpha - 1j * dt * V / (2 * hbar)) * np.ones(N)) + np.diag(alpha * np.ones(N - 1), 1) + np.diag(alpha * np.ones(N - 1), -1)
    return A, B, V

# Initialize wave functions for all cases
psi_all = [psi.copy() for _ in V0_values]
A_all = []
B_all = []
for V0 in V0_values:
    A, B, V = setup_matrices(V0)
    A_all.append(A)
    B_all.append(B)

# Function to calculate transmission coefficient
def calculate_transmission_coefficient(psi):
    return np.sum(np.abs(psi[x >= a]) ** 2) * dx

# Animation Setup
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(-2, 100)
lines = [ax.plot([], [], lw=2, color=color, label=f"$V_0 = {V0}$")[0] for V0, color in zip(V0_values, colors)]
step_line = ax.axvline(x=a, color='black', linestyle='--', linewidth=3, label="Step Position $x = a$")  # Thicker step line
transmission_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, verticalalignment='top')  # Text for transmission coefficients
ax.set_title("Wave Packet Dynamics for Different Step Heights $V_0$")
ax.set_xlabel("Position $x$")
ax.set_ylabel("Probability Density $|\psi|^2$")
ax.legend()
ax.grid()

# Animation Function
def animate(i):
    transmission_coefficients = []
    for j in range(len(V0_values)):
        b = B_all[j] @ psi_all[j]
        psi_all[j] = np.linalg.solve(A_all[j], b)
        lines[j].set_data(x, np.abs(psi_all[j]) ** 2)
        T = calculate_transmission_coefficient(psi_all[j])
        transmission_coefficients.append(f"$V_0 = {V0_values[j]}$: $T = {T:.3f}$")
    transmission_text.set_text("\n".join(transmission_coefficients))  # Update transmission coefficients
    return lines + [step_line, transmission_text]

# Creating and saving animation
output_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), "results", "1D_results")
os.makedirs(output_dir, exist_ok=True)  # Ensure the 1D_results directory exists

output_file = os.path.join(output_dir, "1D-step-potential.mp4")  # Save path
ani = FuncAnimation(fig, animate, frames=steps, interval=50, blit=True)  # Increased interval for slower animation
writer = FFMpegWriter(fps=15, bitrate=1800)  # Reduced fps for slower animation
ani.save(output_file, writer=writer)

#plt.show()

