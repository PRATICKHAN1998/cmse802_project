"""
Module for solving the Crank-Nicolson method in one dimension for a double-well potential.

This script simulates the time evolution of a Gaussian wave packet 
inside a double-well potential using the Crank-Nicolson method. 
The potential consists of two wells separated by a barrier, allowing 
for quantum tunneling between them.

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
L = 4.0    # Length of the domain
N = 500   # Number of spatial points
dx = L / (N - 1)  # Spatial step
x = np.linspace(-L/2, L/2, N)  # Centered around 0
dt = 0.001 # Time step
T_total = 2.0   
steps = int(T_total / dt) 

# Double Well Potential Parameters
V0 = 4000.0  
a = 1.3     # Width and separation of the wells
V = V0 * (x**4 / a**4 - x**2 / a**2)  # Double well potential

# Normalizing the potential for better visualization
V_normalized = V / np.max(np.abs(V)) * 10  # Scale to match the plot range

# Plotting the Double Well Potential
plt.figure()
plt.plot(x, V_normalized, label="Double Well Potential $V(x)$")
plt.title("Double Well Potential")
plt.xlabel("Position $x$")
plt.ylabel("Potential $V(x)$")
plt.legend()
plt.grid()
plt.show()

# Initial Gaussian wave packet
x0 = -1.0       # Initial position (left well)
sigma = 0.1     # Width of the wave packet
k0 = 10.0       # Initial wavenumber
psi = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)  # Normalize

# Crank-Nicolson Coefficients
alpha = 1j * hbar * dt / (2 * m * dx ** 2)
diag = (1 + 2 * alpha + 1j * dt * V / (2 * hbar)) * np.ones(N)
off_diag = -alpha * np.ones(N - 1)
A = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
B = np.diag((1 - 2 * alpha - 1j * dt * V / (2 * hbar)) * np.ones(N)) + np.diag(alpha * np.ones(N - 1), 1) + np.diag(alpha * np.ones(N - 1), -1)

# Animation Setup
fig, ax = plt.subplots()
ax.set_xlim(-L/2, L/2)
ax.set_ylim(-2, 10)
line, = ax.plot([], [], lw=2, label="Probability Density $|\psi|^2$")
ax.plot(x, V_normalized, 'k--', label="Double Well Potential $V(x)$ (scaled)")
ax.set_title("Wave Packet Dynamics in Double Well Potential")
ax.set_xlabel("Position $x$")
ax.set_ylabel("Probability Density $|\psi|^2$")
ax.legend()
ax.grid()

# Animation Function
def animate(i):
    global psi
    b = B @ psi
    psi = np.linalg.solve(A, b)
    line.set_data(x, np.abs(psi) ** 2)
    return line,

# Creating and saving animation
output_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), "results", "1D_results")
os.makedirs(output_dir, exist_ok=True)  

output_file = os.path.join(output_dir, "1D-double-well-potential.mp4")  
ani = FuncAnimation(fig, animate, frames=steps, interval=50, blit=True)  
writer = FFMpegWriter(fps=15, bitrate=1800)  
ani.save(output_file, writer=writer)


