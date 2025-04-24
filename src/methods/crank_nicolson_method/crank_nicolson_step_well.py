"""
1D and 2D simulation of time evolution in step potential wells
using the Crank-Nicolson method.

Features
--------
- Initializes Gaussian wave packets in 1D/2D step potentials
- Evolves wavefunctions using Crank-Nicolson with potential terms
- Visualizes:
    * 1D: Probability density with step position marked
    * 2D: Heatmap and 3D surface plots with potential boundary
- Saves animations as MP4 files in results directory

Author: Agnibha Hanra  
Date: March 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import os

# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0  # Particle mass
L = 1.0  # System length


def initialize_1d_step_potential(N=500, dt=0.001, T_total=0.5, V0=20):
    """
    Initialize the 1D system with a step potential for Crank-Nicolson simulation.

    Parameters
    ----------
    N : int
        Number of spatial grid points
    dt : float
        Time step size
    T_total : float
        Total simulation time
    V0 : float
        Height of the potential step

    Returns
    -------
    x : ndarray
        1D spatial grid
    psi : ndarray
        Initial wave function
    A : ndarray
        Left-hand side matrix for Crank-Nicolson
    B : ndarray
        Right-hand side matrix for Crank-Nicolson
    dt : float
        Time step
    steps : int
        Number of time steps
    a : float
        Step position
    V0 : float
        Potential step height
    """
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    steps = int(T_total / dt)
    a = L / 2

    V = np.zeros(N)
    V[x >= a] = V0

    x0 = L / 4
    sigma = 0.05
    k0 = 5.0
    psi = np.exp(-((x - x0) ** 2) / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)  # Normalize

    alpha = 1j * hbar * dt / (2 * m * dx**2)
    diag = 1 + 2 * alpha + 1j * dt * V / (2 * hbar)
    off_diag = -alpha * np.ones(N - 1)
    A = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    B = (
        np.diag((1 - 2 * alpha - 1j * dt * V / (2 * hbar)))
        + np.diag(alpha * np.ones(N - 1), 1)
        + np.diag(alpha * np.ones(N - 1), -1)
    )

    return x, psi, A, B, dt, steps, a, V0


def time_step_crank_nicolson_1d_step(psi, A, B):
    """
    Perform a single Crank-Nicolson time step in 1D with step potential.

    Parameters
    ----------
    psi : ndarray
        Current wave function
    A : ndarray
        Left-hand side matrix
    B : ndarray
        Right-hand side matrix

    Returns
    -------
    psi_new : ndarray
        Updated wave function after one time step
    """
    b = B @ psi
    psi_new = np.linalg.solve(A, b)
    psi_new[0] = psi_new[-1] = 0  # Boundary conditions
    return psi_new


def initialize_2d_step_potential(N=500, dt=0.001, T_total=0.5, V0=20):
    """
    Initialize the 2D system with a step potential for Crank-Nicolson simulation.

    Parameters
    ----------
    N : int
        Number of grid points per dimension
    dt : float
        Time step size
    T_total : float
        Total simulation time
    V0 : float
        Height of the potential step

    Returns
    -------
    X : ndarray
        2D X-coordinate grid
    Y : ndarray
        2D Y-coordinate grid
    psi : ndarray
        Initial 2D wave function
    V : ndarray
        Potential energy landscape
    D : ndarray
        Crank-Nicolson operator matrix
    D_inv : ndarray
        Inverse of Crank-Nicolson operator
    dt : float
        Time step
    steps : int
        Number of time steps
    a : float
        Step position
    V0 : float
        Potential step height
    """
    dx = L / (N - 1)
    x = y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    steps = int(T_total / dt)
    a = L / 2

    V = np.zeros((N, N))
    V[:, X[0] >= a] = V0

    x0 = y0 = L / 4
    sigma_x = sigma_y = 0.05
    k0x = k0y = 5.0
    psi = np.exp(
        -((X - x0) ** 2) / (2 * sigma_x**2) - (Y - y0) ** 2 / (2 * sigma_y**2)
    ) * np.exp(1j * (k0x * X + k0y * Y))
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx**2)  # Normalize

    alpha = 1j * hbar * dt / (4 * m * dx**2)
    D_x = (
        np.diag(np.ones(N) * (1 + 2 * alpha))
        + np.diag(np.ones(N - 1) * -alpha, 1)
        + np.diag(np.ones(N - 1) * -alpha, -1)
    )
    D_x_inv = np.linalg.inv(D_x)

    return X, Y, psi, V, D_x, D_x_inv, dt, steps, a, V0


def time_step_crank_nicolson_2d_step(psi, V, D, D_inv, dt):
    """
    Perform a single Crank-Nicolson time step in 2D with step potential.

    Parameters
    ----------
    psi : ndarray
        Current 2D wave function
    V : ndarray
        Potential energy landscape
    D : ndarray
        Crank-Nicolson operator
    D_inv : ndarray
        Inverse of Crank-Nicolson operator
    dt : float
        Time step size

    Returns
    -------
    psi : ndarray
        Updated wave function after one time step
    """
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0  # Boundary conditions
    psi = np.exp(-0.5j * V * dt / hbar) * psi  # Potential half-step
    psi = D_inv @ psi @ D_inv.T  # Kinetic step
    psi = np.exp(-0.5j * V * dt / hbar) * psi  # Potential half-step
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0  # Reapply boundaries
    return psi


def run_1d_step_potential_simulation():
    """
    Run and animate the 1D step potential simulation.

    Saves animation as MP4 in results/crank_nicolson_1D_results/
    """
    x, psi, A, B, dt, steps, a, V0 = initialize_1d_step_potential()

    fig, ax = plt.subplots()
    (line,) = ax.plot(x, np.abs(psi) ** 2, color="blue", lw=2)
    ax.axvline(
        x=a, color="black", linestyle="--", linewidth=3, label=f"Step Position $x = a$"
    )
    ax.set_xlim(0, L)
    ax.set_ylim(-2, 30)
    ax.set_xlabel("Position x")
    ax.set_ylabel("Probability Density")
    ax.set_title("1D Wave Packet Across Potential Step (V₀ = {V0})")
    ax.legend()

    def animate(i):
        nonlocal psi
        psi = time_step_crank_nicolson_1d_step(psi, A, B)
        line.set_ydata(np.abs(psi) ** 2)
        return (line,)

    output_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        "results",
        "crank_nicolson_1D_results",
    )
    os.makedirs(output_dir, exist_ok=True)
    ani = FuncAnimation(fig, animate, frames=min(50, steps), interval=50, blit=True)
    ani.save(
        os.path.join(output_dir, "1D_step_potential_CN.mp4"),
        writer=FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.close()


def run_2d_step_potential_simulation():
    """
    Run and animate the 2D step potential simulation.

    Produces:
    - 2D heatmap of probability density
    - 3D surface plot of probability density
    Saves animation as MP4 in results/crank_nicolson_2D_results/
    """
    X, Y, psi, V, D, D_inv, dt, steps, a, V0 = initialize_2d_step_potential()

    fig = plt.figure(figsize=(14, 6))

    # Left: 2D heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(
        np.abs(psi) ** 2,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=np.max(np.abs(psi) ** 2),
    )
    ax1.axvline(
        x=a, color="black", linestyle="--", linewidth=2, label=f"Step at x = {a:.2f}"
    )
    plt.colorbar(im, ax=ax1, label="Probability Density")
    ax1.set_title("2D Probability Density $|\psi(x, y, t)|^2$")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()

    # Right: 3D surface plot

    ax2 = fig.add_subplot(122, projection="3d")
    surf = ax2.plot_surface(
        X, Y, np.abs(psi) ** 2, cmap="hot", rstride=2, cstride=2, alpha=0.8
    )
    ax2.set_title("3D Probability Density")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Probablity Density")
    ax2.set_zlim(0, np.max(np.abs(psi) ** 2))

    fig.suptitle(f"2D Wave Packet Across Potential Step (V₀ = {V0})", fontsize=14)
    plt.tight_layout()

    def animate(i):
        nonlocal psi, surf
        psi = time_step_crank_nicolson_2d_step(psi, V, D, D_inv, dt)

        im.set_array(np.abs(psi) ** 2)
        im.set_clim(0, np.max(np.abs(psi) ** 2))

        ax2.clear()
        surf = ax2.plot_surface(
            X, Y, np.abs(psi) ** 2, cmap="plasma", rstride=2, cstride=2
        )
        ax2.set_zlim(0, np.max(np.abs(psi) ** 2))
        ax2.set_title("3D Probability Density")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Probability Density")

        return im, surf

    output_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        "results",
        "crank_nicolson_2D_results",
    )
    os.makedirs(output_dir, exist_ok=True)
    ani = FuncAnimation(fig, animate, frames=min(50, steps), interval=50, blit=True)
    ani.save(
        os.path.join(output_dir, "2D_step_potential_CN.mp4"),
        writer=FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.close()


if __name__ == "__main__":
    print("Running 1D simulation...")
    run_1d_step_potential_simulation()

    print("Running 2D simulation...")
    run_2d_step_potential_simulation()

    print("Simulations complete! Check the results directory.")
