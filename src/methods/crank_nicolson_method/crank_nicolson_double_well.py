"""
1D and 2D simulation of time evolution in a double-well potential
using the Crank-Nicolson method.

Features
--------
- Simulates quantum wave packet dynamics in double-well potentials
- Handles both 1D and 2D cases with proper normalization
- Uses Crank-Nicolson method for stable time evolution
- Visualizations include:
    * 1D: Wave packet evolution with potential overlay
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
L = 4.0     # System size (symmetric about origin)


def initialize_1d_double_well(N=500, dt=0.001, T_total=0.5, V0=4000.0, a=0.3):
    """
    Initialize the 1D double-well potential system for simulation.
    
    Parameters
    ----------
    N : int
        Number of spatial grid points
    dt : float
        Time step size
    T_total : float
        Total simulation time
    V0 : float
        Potential well depth parameter
    a : float
        Controls width and separation of wells
        
    Returns
    -------
    x : ndarray
        1D spatial grid (-L/2 to L/2)
    psi : ndarray
        Initial wave function (Gaussian wave packet)
    A : ndarray
        Left-hand side matrix for Crank-Nicolson
    B : ndarray
        Right-hand side matrix for Crank-Nicolson
    V_norm : ndarray
        Normalized potential for visualization
    dt : float
        Time step (same as input)
    steps : int
        Number of time steps
    """
    dx = L / (N - 1)
    x = np.linspace(-L / 2, L / 2, N)
    steps = int(T_total / dt)

    # Double-well potential: V(x) = V0*(x⁴/a⁴ - x²/a²)
    V = V0 * (x**4 / a**4 - x**2 / a**2)
    V_norm = V / np.max(np.abs(V)) * 10  # Scaled for visualization

    # Initial Gaussian wave packet
    x0 = -1.0  # Initial position (left well)
    sigma = 0.05  # Width of wave packet
    k0 = 5.0  # Initial momentum
    psi = np.exp(-((x - x0) ** 2) / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)  # Normalization

    # Crank-Nicolson matrices
    alpha = 1j * hbar * dt / (2 * m * dx**2)
    diag = 1 + 2 * alpha + 1j * dt * V / (2 * hbar)
    off_diag = -alpha * np.ones(N - 1)
    A = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    B = (
        np.diag((1 - 2 * alpha - 1j * dt * V / (2 * hbar)))
        + np.diag(alpha * np.ones(N - 1), 1)
        + np.diag(alpha * np.ones(N - 1), -1)
    )

    return x, psi, A, B, V_norm, dt, steps


def time_step_crank_nicolson_1d_double(psi, A, B):
    """
    Perform a single Crank-Nicolson time step in 1D double-well.
    
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
    return psi_new


def initialize_2d_double_well(N=500, dt=0.001, T_total=1.0, V0=4000.0, a=0.3):
    """
    Initialize the 2D double-well potential system for simulation.
    
    Parameters
    ----------
    N : int
        Number of grid points per dimension
    dt : float
        Time step size
    T_total : float
        Total simulation time
    V0 : float
        Potential well depth parameter
    a : float
        Controls width and separation of wells
        
    Returns
    -------
    X : ndarray
        2D X-coordinate grid
    Y : ndarray
        2D Y-coordinate grid
    psi : ndarray
        Initial 2D wave function
    V : ndarray
        2D potential energy landscape
    D : ndarray
        Crank-Nicolson operator matrix
    D_inv : ndarray
        Inverse of Crank-Nicolson operator
    dt : float
        Time step
    steps : int
        Number of time steps
    """
    dx = L / (N - 1)
    x = y = np.linspace(-L / 2, L / 2, N)
    X, Y = np.meshgrid(x, y)
    steps = int(T_total / dt)

    # 2D double-well potential
    V = V0 * ((X**4 + Y**4) / a**4 - (X**2 + Y**2) / a**2)

    # Initial 2D Gaussian wave packet
    x0, y0 = -1.0, 0.0  # Initial position (left well)
    sigma_x = sigma_y = 0.05  # Widths
    k0x, k0y = 5.0, 0.0  # Initial momentum (x-direction only)
    psi = np.exp(
        -((X - x0) ** 2) / (2 * sigma_x**2) - (Y - y0) ** 2 / (2 * sigma_y**2)
    ) * np.exp(1j * (k0x * X + k0y * Y))
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx**2)  # Normalization

    # Crank-Nicolson matrices
    alpha = 1j * hbar * dt / (4 * m * dx**2)
    D = (
        np.diag(np.ones(N) * (1 + 2 * alpha))
        + np.diag(np.ones(N - 1) * -alpha, 1)
        + np.diag(np.ones(N - 1) * -alpha, -1)
    )
    D_inv = np.linalg.inv(D)

    return X, Y, psi, V, D, D_inv, dt, steps


def time_step_crank_nicolson_2d_double(psi, V, D, D_inv, dt):
    """
    Perform a single Crank-Nicolson time step in 2D double-well.
    
    Uses operator splitting with potential term handled separately.
    
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
    # Boundary conditions
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0
    
    # Operator splitting
    psi = np.exp(-0.5j * V * dt / hbar) * psi  # Potential half-step
    psi = D_inv @ psi @ D_inv.T  # Kinetic step
    psi = np.exp(-0.5j * V * dt / hbar) * psi  # Potential half-step
    
    # Reapply boundaries
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0
    return psi


def run_1d_double_well_simulation():
    """
    Run and animate the 1D double-well potential simulation.
    
    Produces:
    - Plot of probability density evolving in time
    - Overlay of the double-well potential
    - Saves animation as MP4 in results/crank_nicolson_1D_results/
    """
    x, psi, A, B, V_norm, dt, steps = initialize_1d_double_well()
    
    fig, ax = plt.subplots()
    (line,) = ax.plot(x, np.abs(psi) ** 2, lw=2)
    ax.plot(x, V_norm, "k--", label="Double Well Potential $V(x)$ (scaled)")
    ax.set_xlim(-L / 2, L / 2)
    ax.set_ylim(-2, 10)
    ax.set_title("1D Quantum Tunneling in Double Well")
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Probability Density $|\psi|^2$")
    ax.legend()

    def animate(i):
        nonlocal psi
        psi = time_step_crank_nicolson_1d_double(psi, A, B)
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
        os.path.join(output_dir, "1D_double_well_CN.mp4"),
        writer=FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.close()


def run_2d_double_well_simulation():
    """
    Run and animate the 2D double-well potential simulation.
    
    Produces:
    - 2D heatmap of probability density
    - 3D surface plot of probability density
    - Saves animation as MP4 in results/crank_nicolson_2D_results/
    """
    X, Y, psi, V, D, D_inv, dt, steps = initialize_2d_double_well()

    fig = plt.figure(figsize=(14, 6))

    # Left: 2D heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(
        np.abs(psi) ** 2,
        extent=[-L / 2, L / 2, -L / 2, L / 2],
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=np.max(np.abs(psi) ** 2),
    )
    plt.colorbar(im, ax=ax1, label="Probability Density")
    ax1.set_title("2D Probability Density $|\psi(x, y, t)|^2$")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Right: 3D surface plot
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf = ax2.plot_surface(
        X, Y, np.abs(psi) ** 2, cmap="hot", rstride=2, cstride=2, antialiased=False
    )
    ax2.set_title("3D Probability Density")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Density")

    fig.suptitle("2D Quantum Tunneling in Double Well", fontsize=14)
    plt.tight_layout()

    def animate(i):
        nonlocal psi, surf
        psi = time_step_crank_nicolson_2d_double(psi, V, D, D_inv, dt)

        # Update 2D heatmap
        im.set_array(np.abs(psi) ** 2)
        im.set_clim(0, np.max(np.abs(psi) ** 2))

        # Update 3D surface
        ax2.clear()
        surf = ax2.plot_surface(
            X, Y, np.abs(psi) ** 2, cmap="hot", rstride=2, cstride=2, antialiased=False
        )
        ax2.set_zlim(0, np.max(np.abs(psi) ** 2))
        ax2.set_title("3D Probability Density")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Density")

        return im, surf

    output_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        "results",
        "crank_nicolson_2D_results",
    )
    os.makedirs(output_dir, exist_ok=True)
    ani = FuncAnimation(fig, animate, frames=min(50, steps), interval=50, blit=True)
    ani.save(
        os.path.join(output_dir, "2D_double_well_CN.mp4"),
        writer=FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.close()


if __name__ == "__main__":
    print("Running 1D simulation...")
    run_1d_double_well_simulation()

    print("Running 2D simulation...")
    run_2d_double_well_simulation()

    print("Simulations complete! Check the results directory.")
