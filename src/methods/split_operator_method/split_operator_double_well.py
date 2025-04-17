"""
1D and 2D simulation of quantum tunneling in double-well potentials
using the Split-Operator Method with Fast Fourier Transforms.

Author: Agnibha Hanra
Date: March 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import os

# Physical constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0  # Mass of the particle


def initialize_1d_system(L=4.0, N=500, dt=0.001, T_total=0.5):
    """Initialize 1D double-well system"""
    dx = L / (N - 1)
    x = np.linspace(-L / 2, L / 2, N)
    steps = int(T_total / dt)

    # Double-well potential
    V0 = 4000.0
    a = 1.3
    V = V0 * (x**4 / a**4 - x**2 / a**2)

    # Wave vector for FFT
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # Initial wave packet (left well)
    x0 = -1.0
    sigma = 0.05
    k0 = 5.0
    psi = np.exp(-((x - x0) ** 2) / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)  # Normalize

    return x, k, psi, V, dt, steps


def initialize_2d_system(L=4.0, N=500, dt=0.001, T_total=0.5):
    """Initialize 2D double-well system"""
    dx = L / (N - 1)
    x = y = np.linspace(-L / 2, L / 2, N)
    X, Y = np.meshgrid(x, y)
    steps = int(T_total / dt)

    # 2D Double-well potential
    V0 = 4000.0
    a = 1.3
    V = V0 * (X**4 / a**4 - X**2 / a**2 + Y**4 / a**4 - Y**2 / a**2)

    # Wave vectors for FFT
    kx = ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    Kx, Ky = np.meshgrid(kx, ky)

    # Initial wave packet (left well)
    x0, y0 = -1.0, 0.0
    sigma_x = sigma_y = 0.05
    k0x, k0y = 5.0, 0.0
    psi = np.exp(
        -((X - x0) ** 2) / (2 * sigma_x**2) - (Y - y0) ** 2 / (2 * sigma_y**2)
    ) * np.exp(1j * (k0x * X + k0y * Y))
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx**2)  # Normalize

    return X, Y, Kx, Ky, psi, V, dt, steps


def time_step_1d(psi, V, k, dt):
    """1D time evolution using split-operator method"""
    psi = np.exp(-0.5j * V * dt / hbar) * psi
    psi_k = np.fft.fft(psi)
    psi_k *= np.exp(-0.5j * hbar * k**2 * dt / m)
    psi = np.fft.ifft(psi_k)
    psi = np.exp(-0.5j * V * dt / hbar) * psi
    return psi


def time_step_2d(psi, V, Kx, Ky, dt):
    """2D time evolution using split-operator method"""
    psi = np.exp(-0.5j * V * dt / hbar) * psi
    psi_k = np.fft.fft2(psi)
    psi_k *= np.exp(-0.5j * hbar * (Kx**2 + Ky**2) * dt / m)
    psi = np.fft.ifft2(psi_k)
    psi = np.exp(-0.5j * V * dt / hbar) * psi
    return psi


def run_1d_double_well_simulation():
    """Run and visualize 1D tunneling simulation"""
    x, k, psi, V, dt, steps = initialize_1d_system()

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 6))
    (line,) = ax.plot(x, np.abs(psi) ** 2, "b-", lw=2, label="$|\psi|^2$")
    (pot_line,) = ax.plot(x, V / 400, "k--", label="Potential (scaled)")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2.10)
    ax.set_title("1D Quantum Tunneling in Double Well")
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    def animate(i):
        nonlocal psi
        psi = time_step_1d(psi, V, k, dt)
        line.set_ydata(np.abs(psi) ** 2)
        return line, pot_line

    # Save animation
    output_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        "results",
        "split_operator_1D_results",
    )
    os.makedirs(output_dir, exist_ok=True)
    ani = FuncAnimation(fig, animate, frames=min(50, steps), interval=50, blit=True)
    ani.save(
        os.path.join(output_dir, "1D_double_well_SplitOperator.mp4"),
        writer=FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.close()


def run_2d_double_well_simulation():
    """Run and visualize 2D tunneling simulation"""
    X, Y, Kx, Ky, psi, V, dt, steps = initialize_2d_system()

    # First save potential plot
    fig_pot = plt.figure(figsize=(10, 8))
    ax_pot = fig_pot.add_subplot(111, projection="3d")
    surf = ax_pot.plot_surface(X, Y, V, cmap="viridis", rstride=2, cstride=2)
    fig_pot.colorbar(surf, ax=ax_pot, label="Potential Energy")
    ax_pot.set_title("2D Double-Well Potential")
    output_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        "results",
        "split_operator_2D_results",
    )

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "2D_potential.png"))
    plt.close(fig_pot)

    # Setup animation
    fig = plt.figure(figsize=(14, 6))

    # 2D Density plot
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(
        np.abs(psi) ** 2, extent=[-2, 2, -2, 2], origin="lower", cmap="viridis"
    )
    plt.colorbar(im, ax=ax1, label="Probability Density")
    ax1.set_title("2D Probability Density")

    # 3D Surface plot
    ax2 = fig.add_subplot(122, projection="3d")
    surf = ax2.plot_surface(
        X, Y, np.abs(psi) ** 2, cmap="hot", rstride=2, cstride=2, alpha=0.8
    )
    ax2.set_title("3D Probability Density")

    fig.suptitle("2D Quantum Tunneling in Double Well", fontsize=14)
    plt.tight_layout()

    def animate(i):
        nonlocal psi
        psi = time_step_2d(psi, V, Kx, Ky, dt)

        # Update 2D plot
        im.set_array(np.abs(psi) ** 2)
        im.set_clim(0, np.max(np.abs(psi) ** 2))

        # Update 3D plot
        ax2.clear()
        surf = ax2.plot_surface(
            X, Y, np.abs(psi) ** 2, cmap="hot", rstride=2, cstride=2, alpha=0.8
        )
        ax2.set_zlim(0, np.max(np.abs(psi) ** 2))
        ax2.set_title("3D Probability Density")

        return im, surf

    # Save animation
    output_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        "results",
        "split_operator_2D_results",
    )
    ani = FuncAnimation(fig, animate, frames=min(50, steps), interval=50, blit=True)
    ani.save(
        os.path.join(output_dir, "2D_double_well_SplitOperator.mp4"),
        writer=FFMpegWriter(fps=10, bitrate=1800),
    )
    plt.close()


if __name__ == "__main__":
    print("Running 1D simulation...")
    run_1d_double_well_simulation()

    print("Running 2D simulation...")
    run_2d_double_well_simulation()

    print("Simulations complete! Check the results directory.")
