"""
1D and 2D Quantum Wave Packet Simulation using the Crank-Nicolson Method
=========================================================================

This module implements numerical simulations of a Gaussian wave packet
in infinite square potential wells in both 1D and 2D using the Crank-Nicolson method.

Features
--------
- Initializes a normalized wave packet in 1D and 2D.
- Evolves the system in time using the Crank-Nicolson scheme.
- Produces animations:
    * 1D: Probability density |ψ(x, t)|²
    * 2D: Heatmap and 3D surface of |ψ(x, y, t)|²
- Saves results in structured directories under "results/".

Author: Agnibha Hanra  
Date: March 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import os

# Constants
hbar = 1.0
m = 1.0
L = 1.0


def initialize_1d_system(N=500, dt=0.001, T_total=0.5):
    """
    Initialize the 1D wave packet system for Crank-Nicolson simulation.

    Parameters
    ----------
    N : int
        Number of spatial grid points.
    dt : float
        Time step size.
    T_total : float
        Total simulation time.

    Returns
    -------
    x : ndarray
        1D spatial grid.
    psi : ndarray
        Initial normalized wave function.
    A : ndarray
        Left-hand side matrix for Crank-Nicolson method.
    B : ndarray
        Right-hand side matrix for Crank-Nicolson method.
    dt : float
        Time step (returned again for convenience).
    steps : int
        Number of time steps.
    """
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    steps = int(T_total / dt)

    x0 = L / 2
    sigma = 0.05
    k0 = 5.0
    psi = np.exp(-((x - x0) ** 2) / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)  # Normalize

    alpha = 1j * hbar * dt / (2 * m * dx**2)
    diag = (1 + 2 * alpha) * np.ones(N)
    off_diag = -alpha * np.ones(N - 1)
    A = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    B = (
        np.diag((1 - 2 * alpha) * np.ones(N))
        + np.diag(alpha * np.ones(N - 1), 1)
        + np.diag(alpha * np.ones(N - 1), -1)
    )

    return x, psi, A, B, dt, steps


def time_step_crank_nicolson_1d(psi, A, B):
    """
    Perform a single Crank-Nicolson time step in 1D.

    Parameters
    ----------
    psi : ndarray
        Current wave function.
    A : ndarray
        Left-hand side matrix.
    B : ndarray
        Right-hand side matrix.

    Returns
    -------
    psi_new : ndarray
        Updated wave function after one time step.
    """
    b = B @ psi
    psi_new = np.linalg.solve(A, b)
    psi_new[0] = psi_new[-1] = 0
    return psi_new


def initialize_2d_system(N=500, dt=0.001, T_total=0.5):
    """
    Initialize the 2D wave packet system for Crank-Nicolson simulation.

    Parameters
    ----------
    N : int
        Number of spatial grid points per dimension.
    dt : float
        Time step size.
    T_total : float
        Total simulation time.

    Returns
    -------
    X : ndarray
        2D X-coordinate meshgrid.
    Y : ndarray
        2D Y-coordinate meshgrid.
    psi : ndarray
        Initial normalized 2D wave function.
    D : ndarray
        Crank-Nicolson matrix for implicit solve.
    D_inv : ndarray
        Inverse of matrix D.
    dt : float
        Time step.
    steps : int
        Number of time steps.
    """
    dx = L / (N - 1)
    x = y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    steps = int(T_total / dt)

    x0 = y0 = L / 2
    sigma_x = sigma_y = 0.05
    k0x = k0y = 5.0
    psi = np.exp(
        -((X - x0) ** 2) / (2 * sigma_x**2) - (Y - y0) ** 2 / (2 * sigma_y**2)
    ) * np.exp(1j * (k0x * X + k0y * Y))
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx**2)  # Normalize

    alpha = 1j * hbar * dt / (4 * m * dx**2)
    I = np.eye(N)
    D = (
        np.diag(np.ones(N) * (1 + 2 * alpha))
        + np.diag(np.ones(N - 1) * -alpha, 1)
        + np.diag(np.ones(N - 1) * -alpha, -1)
    )
    D_inv = np.linalg.inv(D)

    return X, Y, psi, D, D_inv, dt, steps


def time_step_crank_nicolson_2d(psi, D, D_inv):
    """
    Perform a single Crank-Nicolson time step in 2D using ADI (Alternating Direction Implicit) method.

    Parameters
    ----------
    psi : ndarray
        Current 2D wave function.
    D : ndarray
        Crank-Nicolson operator.
    D_inv : ndarray
        Inverse of Crank-Nicolson operator.

    Returns
    -------
    psi : ndarray
        Updated wave function after one time step.
    """
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0
    psi = D_inv @ psi @ D_inv.T
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0
    return psi


def run_1d_simulation():
    """
    Run the 1D wave packet simulation and save the animation to MP4.

    Returns
    -------
    None
    """
    x, psi, A, B, dt, steps = initialize_1d_system()
    fig, ax = plt.subplots()
    (line,) = ax.plot(x, np.abs(psi) ** 2)
    ax.set_xlim(0, L)
    ax.set_ylim(0, np.max(np.abs(psi) ** 2) * 1.1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability Density")
    ax.set_title("1D Infinite Square Well Wave Packet Dynamics")

    def animate(i):
        nonlocal psi
        psi = time_step_crank_nicolson_1d(psi, A, B)
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
        os.path.join(output_dir, "1D_infinite_square_potential_CN.mp4"),
        writer=FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.close()


def run_2d_simulation():
    """
    Run the 2D wave packet simulation and save the animation as MP4, including:
    - 2D heatmap of |ψ(x,y,t)|²
    - 3D surface plot of |ψ(x,y,t)|²

    Returns
    -------
    None
    """
    X, Y, psi, D, D_inv, dt, steps = initialize_2d_system()

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
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
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf = ax2.plot_surface(
        X, Y, np.abs(psi) ** 2, cmap="hot", rstride=2, cstride=2, antialiased=False
    )
    ax2.set_title("3D Probability Density")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Probablity Density")
    fig.suptitle("2D Infinite Square Well Wave Packet Dynamics", fontsize=14)
    plt.tight_layout()

    def animate(i):
        nonlocal psi, surf
        psi = time_step_crank_nicolson_2d(psi, D, D_inv)

        im.set_array(np.abs(psi) ** 2)
        im.set_clim(0, np.max(np.abs(psi) ** 2))

        ax2.clear()
        surf = ax2.plot_surface(
            X, Y, np.abs(psi) ** 2, cmap="hot", rstride=2, cstride=2, antialiased=False
        )
        ax2.set_zlim(0, np.max(np.abs(psi) ** 2))
        ax2.set_title("3D Probability Density")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel(" Probablity Density")

        return im, surf

    output_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        "results",
        "crank_nicolson_2D_results",
    )
    os.makedirs(output_dir, exist_ok=True)
    ani = FuncAnimation(fig, animate, frames=min(50, steps), interval=50, blit=True)
    ani.save(
        os.path.join(output_dir, "2D_infinite_square_potential_CN.mp4"),
        writer=FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.close()


if __name__ == "__main__":
    """
    Execute 1D and 2D simulations if the module is run as a script.
    """
    print("Running 1D simulation...")
    run_1d_simulation()

    print("Running 2D simulation...")
    run_2d_simulation()

    print("Simulations complete! Check the results directory.")
