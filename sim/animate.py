"""Renders a cart-pole state trajectory as an animated GIF using
matplotlib's PillowWriter (no ffmpeg dependency needed)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from sim.dynamics import CartPoleParams


def render_animation(
    times: np.ndarray,
    states: np.ndarray,
    params: CartPoleParams,
    title: str,
    save_path: str,
    fps: int = 50,
    stride: int = 1,
):
    cart_w, cart_h = 0.4, 0.2
    pole_len = 2 * params.pole_half_length

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-params.track_limit - 0.5, params.track_limit + 0.5)
    ax.set_ylim(-0.5, pole_len + 0.6)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes)

    cart_patch = Rectangle((0, 0), cart_w, cart_h, fc="#2a6fdb", ec="black")
    ax.add_patch(cart_patch)
    (pole_line,) = ax.plot([], [], lw=4, color="#d6482f", solid_capstyle="round")
    (bob,) = ax.plot([], [], "o", color="#7a1f0d", markersize=8)

    frames = range(0, len(times), stride)

    def init():
        cart_patch.set_xy((-cart_w / 2, -cart_h / 2))
        pole_line.set_data([], [])
        bob.set_data([], [])
        time_text.set_text("")
        return cart_patch, pole_line, bob, time_text

    def update(i):
        x, _, theta, _ = states[i]
        cart_patch.set_xy((x - cart_w / 2, -cart_h / 2))
        pivot_x, pivot_y = x, cart_h / 2
        tip_x = pivot_x + pole_len * np.sin(theta)
        tip_y = pivot_y + pole_len * np.cos(theta)
        pole_line.set_data([pivot_x, tip_x], [pivot_y, tip_y])
        bob.set_data([tip_x], [tip_y])
        time_text.set_text(f"t = {times[i]:.2f} s   theta = {np.degrees(theta):.1f} deg")
        return cart_patch, pole_line, bob, time_text

    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
