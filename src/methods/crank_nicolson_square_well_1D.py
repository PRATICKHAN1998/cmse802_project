"""
Module for solving the Crank-Nicolson method in one dimension for 
an infinite square potential well.

This script simulates the time evolution of a Gaussian wave packet 
inside an infinite square well using the Crank-Nicolson method.

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
L = 1.0    # Length of the infinite square well
N = 500   # Number of spatial points
dx = L / (N - 1)  # Spatial step
x = np.linspace(0, L, N)
dt = 0.001 # Time step
T_total = 1.0   
steps = int(T_total / dt)  

# Initial Gaussian wave packet
x0 = L / 4       # Initial position (left of the well center)
sigma = 0.05     # Width of the wave packet
k0 = 50.0        # Initial wavenumber
psi = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)  # Normalize

# Crank-Nicholson Coefficients
alpha = 1j * hbar * dt / (2 * m * dx ** 2)
diag = (1 + 2 * alpha) * np.ones(N)
off_diag = -alpha * np.ones(N - 1)
A = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
B = np.diag((1 - 2 * alpha) * np.ones(N)) + np.diag(alpha * np.ones(N - 1), 1) + np.diag(alpha * np.ones(N - 1), -1)

def time_step(psi):
    """
    Perform one time step using the Crank-Nicolson method.
    
    Parameters:
    psi (numpy.ndarray): Current wave function values.
    
    Returns:
    numpy.ndarray: Updated wave function after one time step.
    """
    b = B @ psi
    psi_new = np.linalg.solve(A, b)
    psi_new[0] = psi_new[-1] = 0  # Boundary conditions for infinite well
    return psi_new

# Animation Setup
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, 10)
line, = ax.plot([], [], lw=2)
ax.set_title("Gaussian Wave Packet in Infinite Square Well")
ax.set_xlabel("Position")
ax.set_ylabel("Probability Density")

def animate(i):
    """
    Update the animation frame by evolving the wave function.
    
    Parameters:
    i (int): Frame index.
    
    Returns:
    tuple: Updated plot line.
    """
    global psi
    psi = time_step(psi)
    line.set_data(x, np.abs(psi) ** 2)
    return line,

# Creating and saving animation
output_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), "results", "1D_results")
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "1D-infinite-square-potential.mp4")  
ani = FuncAnimation(fig, animate, frames=steps, interval=50, blit=True)  
writer = FFMpegWriter(fps=15, bitrate=1800)  
ani.save(output_file, writer=writer)

