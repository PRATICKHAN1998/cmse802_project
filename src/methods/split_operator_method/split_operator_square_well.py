"""
1D and 2D simulation of time evolution in infinite square potential wells
using the Split-Operator Method with Fast Fourier Transforms.

Features
--------
- Simulates quantum wave packet dynamics in infinite square wells
- Uses Split-Operator Method with FFT for efficient time evolution
- Handles both 1D and 2D cases with proper normalization
- Visualizations include:
    * 1D: Probability density evolution
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
m = 1.0     # Particle mass (natural units)
L = 1.0     # Length of the infinite square well


def initialize_1d_system(N=500, dt=0.001, T_total=0.5):
    """
    Initialize the 1D infinite square well system for simulation.
    
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
    kx : ndarray
        Wave vector grid for FFT
    psi : ndarray
        Initial wave function (Gaussian wave packet)
    V : ndarray
        Potential energy profile (infinite square well)
    dt : float
        Time step (same as input)
    steps : int
        Number of time steps
    """
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    steps = int(T_total / dt)

    # Wave vectors for FFT (momentum space representation)
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # Initial Gaussian wave packet
    x0 = L / 2  # Center of well
    sigma = 0.05  # Width of wave packet
    k0 = 5.0  # Initial momentum
    psi = np.exp(-((x - x0) ** 2) / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)  # Normalization

    # Infinite square well potential
    V = np.zeros(N)
    V[0] = V[-1] = 1e10  # Infinite potential at boundaries

    return x, kx, psi, V, dt, steps


def initialize_2d_system(N=500, dt=0.001, T_total=0.5):
    """
    Initialize the 2D infinite square well system for simulation.
    
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
        2D potential energy landscape (infinite square well)
    dt : float
        Time step
    steps : int
        Number of time steps
    """
    dx = L / (N - 1)
    x = y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    steps = int(T_total / dt)

    # Wave vectors for 2D FFT
    kx = ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    Kx, Ky = np.meshgrid(kx, ky)

    # Initial 2D Gaussian wave packet
    x0 = y0 = L / 2  # Center of well
    sigma_x = sigma_y = 0.05  # Widths
    k0x = k0y = 5.0  # Initial momentum
    psi = np.exp(
        -((X - x0) ** 2) / (2 * sigma_x**2) - (Y - y0) ** 2 / (2 * sigma_y**2)
    ) * np.exp(1j * (k0x * X + k0y * Y))
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx**2)  # Normalization

    # 2D infinite square well potential
    V = np.zeros((N, N))
    V[0, :] = V[-1, :] = V[:, 0] = V[:, -1] = 1e10  # Infinite potential at boundaries

    return X, Y, Kx, Ky, psi, V, dt, steps


def time_step_split_operator_1d(psi, V, kx, dt):
    """
    Perform a single time step in 1D using Split-Operator Method.
    
    Parameters
    ----------
    psi : ndarray
        Current wave function
    V : ndarray
        Potential energy profile
    kx : ndarray
        Wave vector grid
    dt : float
        Time step size
        
    Returns
    -------
    psi : ndarray
        Updated wave function after one time step
    """
    # Half-step in position space
    psi = np.exp(-0.5j * V * dt / hbar) * psi

    # Full-step in momentum space
    psi_k = np.fft.fft(psi)
    psi_k *= np.exp(-0.5j * hbar * kx**2 * dt / m)
    psi = np.fft.ifft(psi_k)

    # Another half-step in position space
    psi = np.exp(-0.5j * V * dt / hbar) * psi

    # Enforce boundary conditions
    psi[0] = psi[-1] = 0

    return psi


def time_step_split_operator_2d(psi, V, Kx, Ky, dt):
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
    # Half-step in position space
    psi = np.exp(-0.5j * V * dt / hbar) * psi

    # Full-step in momentum space
    psi_k = np.fft.fft2(psi)
    psi_k *= np.exp(-0.5j * hbar * (Kx**2 + Ky**2) * dt / m)
    psi = np.fft.ifft2(psi_k)

    # Another half-step in position space
    psi = np.exp(-0.5j * V * dt / hbar) * psi

    # Enforce boundary conditions
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0

    return psi


def run_1d_square_potential_simulation():
    """
    Run and animate the 1D infinite square well simulation.
    
    Produces:
    - Plot of probability density evolving in time
    - Saves animation as MP4 in results/split_operator_1D_results/
    """
    x, kx, psi, V, dt, steps = initialize_1d_system(N=500)

    # Setup 1D plot
    fig, ax = plt.subplots(figsize=(10, 6))
    (line,) = ax.plot(x, np.abs(psi) ** 2, "b-", linewidth=2)
    ax.set_xlim(0, L)
    ax.set_ylim(0, np.max(np.abs(psi) ** 2 * 1.1))
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability Density")
    ax.set_title("1D Infinite Square Well Wave Packet Dynamics")

    def animate(i):
        nonlocal psi
        psi = time_step_split_operator_1d(psi, V, kx, dt)
        line.set_ydata(np.abs(psi) ** 2)
        ax.set_ylim(0, np.max(np.abs(psi) ** 2 * 1.1))
        return (line,)

    # Create and save animation
    output_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        "results",
        "split_operator_1D_results",
    )
    os.makedirs(output_dir, exist_ok=True)
    ani = FuncAnimation(fig, animate, frames=min(50, steps), interval=50, blit=True)
    ani.save(
        os.path.join(output_dir, "1D_infinite_square_potential_SplitOperator.mp4"),
        writer=FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.close()


def run_2d_square_potential_simulation():
    """
    Run and animate the 2D infinite square well simulation.
    
    Produces:
    - 2D heatmap of probability density
    - 3D surface plot of probability density
    - Saves animation as MP4 in results/split_operator_2D_results/
    """
    X, Y, Kx, Ky, psi, V, dt, steps = initialize_2d_system(N=500)

    # Setup 2D plots
    fig = plt.figure(figsize=(14, 6))

    # 1. 2D Density Plot
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(
        np.abs(psi) ** 2,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=np.max(np.abs(psi) ** 2),
    )
    plt.colorbar(im, ax=ax1, label="Probability Density")
    ax1.set_title("2D Probability Density $|\psi(x,y,t)|^2$")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")

    # 2. 3D Surface Plot
    ax2 = fig.add_subplot(122, projection="3d")
    surf = ax2.plot_surface(
        X, Y, np.abs(psi) ** 2, cmap="hot", rstride=2, cstride=2, antialiased=False
    )
    ax2.set_title("3D Probability Density")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.set_zlabel("Probability Density")

    fig.suptitle("2D Infinite Square Well Wave Packet Dynamics", fontsize=14)
    plt.tight_layout()

    def animate(i):
        nonlocal psi, surf
        psi = time_step_split_operator_2d(psi, V, Kx, Ky, dt)

        # Update 2D plot
        im.set_array(np.abs(psi) ** 2)
        im.set_clim(0, np.max(np.abs(psi) ** 2))

        # Update 3D plot
        ax2.clear()
        surf = ax2.plot_surface(
            X, Y, np.abs(psi) ** 2, cmap="hot", rstride=2, cstride=2, antialiased=False
        )
        ax2.set_zlim(0, np.max(np.abs(psi) ** 2))
        ax2.set_title("3D Probability Density")
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Y Position")
        ax2.set_zlabel("Probability Density")

        return im, surf

    # Create and save animation
    output_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        "results",
        "split_operator_2D_results",
    )
    os.makedirs(output_dir, exist_ok=True)
    ani = FuncAnimation(fig, animate, frames=min(50, steps), interval=50, blit=False)
    ani.save(
        os.path.join(output_dir, "2D_infinite_square_potential_SplitOperator.mp4"),
        writer=FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.close()


if __name__ == "__main__":
    print("Running 1D simulation...")
    run_1d_square_potential_simulation()

    print("Running 2D simulation...")
    run_2d_square_potential_simulation()

    print("Simulations complete! Check the results directory.")
