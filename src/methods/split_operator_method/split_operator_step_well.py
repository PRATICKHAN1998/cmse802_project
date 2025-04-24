"""
1D and 2D simulation of quantum wave packet dynamics across potential steps
using the Split-Operator Method with Fast Fourier Transforms.

Features
--------
- Simulates quantum wave packet evolution across potential steps
- Uses Split-Operator Method with FFT for efficient time evolution
- Handles both 1D and 2D cases with proper normalization
- Visualizations include:
    * 1D: Probability density evolution with step position marked
    * 2D: Heatmap and 3D surface plots of probability density
- Saves animations as MP4 files in organized directory structure

Author: Agnibha Hanra  
Date: March 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import os

# Physical constants
hbar = 1.0  # Reduced Planck's constant (natural units)
m = 1.0  # Particle mass (natural units)
L = 1.0  # Length of the spatial domain


def initialize_1d_system(N=500, dt=0.001, T_total=0.1):
    """
    Initialize the 1D system with potential step for simulation.

    Parameters
    ----------
    N : int
        Number of spatial grid points
    dt : float
        Time step size
    T_total : float
        Total simulation time

    Returns
    -------
    x : ndarray
        1D spatial grid (0 to L)
    k : ndarray
        Wave vector grid for FFT
    psi : ndarray
        Initial wave function (Gaussian wave packet)
    V : ndarray
        Potential energy profile (step potential)
    dt : float
        Time step (same as input)
    steps : int
        Number of time steps
    a : float
        Position of potential step
    V0 : float
        Height of potential step
    """
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    steps = int(T_total / dt)

    # Wave vector for FFT (momentum space representation)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # Initial Gaussian wave packet
    x0 = L / 4  # Initial position (left of step)
    sigma = 0.05  # Width of wave packet
    k0 = 5.0  # Initial momentum
    psi = np.exp(-((x - x0) ** 2) / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)  # Normalization

    # Potential Step configuration
    a = L / 2  # Step position
    V0 = 20  # Step height
    V = np.zeros(N)
    V[x >= a] = V0  # Create step at x = a

    return x, k, psi, V, dt, steps, a, V0


def initialize_2d_system(N=500, dt=0.001, T_total=0.5):
    """
    Initialize the 2D system with potential step for simulation.

    Parameters
    ----------
    N : int
        Number of grid points per dimension
    dt : float
        Time step size
    T_total : float
        Total simulation time

    Returns
    -------
    X : ndarray
        2D X-coordinate grid
    Y : ndarray
        2D Y-coordinate grid
    Kx : ndarray
        X-component wave vector grid for FFT
    Ky : ndarray
        Y-component wave vector grid for FFT
    psi : ndarray
        Initial 2D wave function
    V : ndarray
        2D potential energy landscape (step potential)
    dt : float
        Time step
    steps : int
        Number of time steps
    a : float
        Position of potential step
    V0 : float
        Height of potential step
    """
    dx = L / (N - 1)
    x = y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    steps = int(T_total / dt)

    # Wave vectors for 2D FFT
    kx = ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    Kx, Ky = np.meshgrid(kx, ky)

    # Initial 2D Gaussian wave packet
    x0 = y0 = L / 4  # Initial position (left of step)
    sigma_x = sigma_y = 0.05  # Widths
    k0x, k0y = 5.0, 0.0  # Initial momentum (x-direction only)
    psi = np.exp(
        -((X - x0) ** 2) / (2 * sigma_x**2) - (Y - y0) ** 2 / (2 * sigma_y**2)
    ) * np.exp(1j * (k0x * X + k0y * Y))
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx**2)  # Normalization

    # 2D Potential Step configuration
    a = L / 2  # Step position
    V0 = 20  # Step height
    V = np.zeros((N, N))
    V[X >= a] = V0  # Create step at x = a

    return X, Y, Kx, Ky, psi, V, dt, steps, a, V0


def time_step_1d(psi, V, k, dt):
    """
    Perform a single time step in 1D using Split-Operator Method.

    Parameters
    ----------
    psi : ndarray
        Current wave function
    V : ndarray
        Potential energy profile
    k : ndarray
        Wave vector grid
    dt : float
        Time step size

    Returns
    -------
    psi : ndarray
        Updated wave function after one time step
    """
    # Split-Operator steps:
    # 1. Half-step in position space
    psi = np.exp(-0.5j * V * dt / hbar) * psi

    # 2. Full-step in momentum space
    psi_k = np.fft.fft(psi)
    psi_k *= np.exp(-0.5j * hbar * k**2 * dt / m)
    psi = np.fft.ifft(psi_k)

    # 3. Another half-step in position space
    psi = np.exp(-0.5j * V * dt / hbar) * psi

    return psi


def time_step_2d(psi, V, Kx, Ky, dt):
    """
    Perform a single time step in 2D using Split-Operator Method.

    Parameters
    ----------
    psi : ndarray
        Current 2D wave function
    V : ndarray
        2D potential energy landscape
    Kx : ndarray
        X-component wave vector grid
    Ky : ndarray
        Y-component wave vector grid
    dt : float
        Time step size

    Returns
    -------
    psi : ndarray
        Updated wave function after one time step
    """
    # Split-Operator steps:
    # 1. Half-step in position space
    psi = np.exp(-0.5j * V * dt / hbar) * psi

    # 2. Full-step in momentum space
    psi_k = np.fft.fft2(psi)
    psi_k *= np.exp(-0.5j * hbar * (Kx**2 + Ky**2) * dt / m)
    psi = np.fft.ifft2(psi_k)

    # 3. Another half-step in position space
    psi = np.exp(-0.5j * V * dt / hbar) * psi

    return psi


def run_1d_step_potential_simulation():
    """
    Run and animate the 1D step potential simulation.

    Produces:
    - Plot of probability density evolving in time
    - Vertical line marking step position
    - Saves animation as MP4 in results/split_operator_1D_results/
    """
    x, k, psi, V, dt, steps, a, V0 = initialize_1d_system()

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 6))
    (line,) = ax.plot(x, np.abs(psi) ** 2, "b-", linewidth=2)
    step_line = ax.axvline(x=a, color="black", linestyle="--", linewidth=3)
    ax.set_xlim(0, L)
    ax.set_ylim(0, 30)
    ax.set_title(f"1D Wave Packet Across Potential Step (V₀ = {V0})")
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability Density")
    ax.grid(True, alpha=0.3)

    def animate(i):
        nonlocal psi
        psi = time_step_1d(psi, V, k, dt)
        line.set_ydata(np.abs(psi) ** 2)
        return line, step_line

    # Save animation
    output_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        "results",
        "split_operator_1D_results",
    )
    os.makedirs(output_dir, exist_ok=True)
    ani = FuncAnimation(fig, animate, frames=min(50, steps), interval=50, blit=True)
    ani.save(
        os.path.join(output_dir, "1D_step_potential_SplitOperator.mp4"),
        writer=FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.close()


def run_2d_step_potential_simulation():
    """
    Run and animate the 2D step potential simulation.

    Produces:
    - 2D heatmap of probability density with step position marked
    - 3D surface plot of probability density
    - Saves animation as MP4 in results/split_operator_2D_results/
    """
    X, Y, Kx, Ky, psi, V, dt, steps, a, V0 = initialize_2d_system()

    # Setup combined figure
    fig = plt.figure(figsize=(14, 6))

    # 2D Density Plot
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(
        np.abs(psi) ** 2, extent=[0, L, 0, L], origin="lower", cmap="viridis"
    )
    ax1.axvline(x=a, color="red", linestyle="--", linewidth=2)
    plt.colorbar(im, ax=ax1, label="Probability Density")
    ax1.set_title("2D Probability Density $|\psi(x,y,t)|^2$")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # 3D Surface Plot
    ax2 = fig.add_subplot(122, projection="3d")
    surf = ax2.plot_surface(
        X, Y, np.abs(psi) ** 2, cmap="hot", rstride=2, cstride=2, alpha=0.8
    )
    ax2.set_title("3D Probability Density")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Probability Density")
    ax2.set_zlim(0, np.max(np.abs(psi) ** 2))

    fig.suptitle(f"2D Wave Packet Across Potential Step (V₀ = {V0})", fontsize=14)
    plt.tight_layout()

    def animate(i):
        nonlocal psi
        psi = time_step_2d(psi, V, Kx, Ky, dt)

        # Update 2D heatmap
        im.set_array(np.abs(psi) ** 2)
        im.set_clim(0, np.max(np.abs(psi) ** 2))

        # Update 3D surface
        ax2.clear()
        surf = ax2.plot_surface(
            X, Y, np.abs(psi) ** 2, cmap="hot", rstride=2, cstride=2, alpha=0.8
        )
        ax2.set_zlim(0, np.max(np.abs(psi) ** 2))
        ax2.set_title("3D Probability Density")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Probablity Density")

        return [im, surf]

    # Save animation
    output_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        "results",
        "split_operator_2D_results",
    )
    os.makedirs(output_dir, exist_ok=True)
    ani = FuncAnimation(fig, animate, frames=min(50, steps), interval=50, blit=False)
    ani.save(
        os.path.join(output_dir, "2D_step_potential_SplitOperator.mp4"),
        writer=FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.close()


if __name__ == "__main__":
    print("Running 1D simulation...")
    run_1d_step_potential_simulation()

    print("Running 2D simulation...")
    run_2d_step_potential_simulation()

    print("Simulations complete! Check the results directory.")
