"""
Hugging Face Space: interactive cart-pole control comparison.

Pick a controller (PID, LQR, or a trained RL/PPO policy), set an initial
pole tilt and an optional mid-run disturbance push, and watch the same
nonlinear cart-pole plant get balanced -- with a live animation, the
state/force traces, and the real numeric recovery metrics.
"""
import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controllers.lqr import LQRController
from controllers.pid import PIDController
from sim.animate import render_animation
from sim.dynamics import CartPoleParams
from sim.experiment import compute_metrics, simulate

HERE = os.path.dirname(__file__)
PARAMS = CartPoleParams()

RL_PATH = os.path.join(HERE, "rl", "ppo_cartpole.zip")
_rl_controller = None
if os.path.exists(RL_PATH):
    from controllers.rl_controller import RLController

    _rl_controller = RLController(RL_PATH, PARAMS)


def build_controller(name: str):
    if name == "PID (angle only)":
        return PIDController(PARAMS)
    if name == "LQR (optimal, full state)":
        return LQRController(PARAMS)
    if name == "RL / PPO (learned)":
        if _rl_controller is None:
            raise gr.Error("No trained RL model is bundled with this Space.")
        return _rl_controller
    raise gr.Error(f"Unknown controller: {name}")


def run(controller_name, initial_theta_deg, apply_disturbance, disturbance_time, disturbance_strength):
    controller = build_controller(controller_name)
    disturbance_t = disturbance_time if apply_disturbance else None

    times, states, forces = simulate(
        controller,
        PARAMS,
        initial_theta=np.radians(initial_theta_deg),
        disturbance_time=disturbance_t,
        disturbance_theta_dot=disturbance_strength,
    )
    m = compute_metrics(times, states, forces, controller_name, disturbance_time=disturbance_time)

    gif_path = os.path.join(HERE, "last_run.gif")
    render_animation(times, states, PARAMS, title=controller_name, save_path=gif_path)

    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    axes[0].plot(times, np.degrees(states[:, 2]))
    axes[0].set_ylabel("pole angle (deg)")
    axes[1].plot(times, states[:, 0])
    axes[1].set_ylabel("cart position (m)")
    axes[2].plot(times[:-1], forces)
    axes[2].set_ylabel("force (N)")
    axes[2].set_xlabel("time (s)")
    if apply_disturbance:
        for ax in axes:
            ax.axvline(disturbance_time, color="gray", linestyle="--", linewidth=1)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    metrics_md = (
        f"**Max |angle|:** {m['max_abs_theta_deg']:.2f} deg &nbsp;&nbsp; "
        f"**Max |cart pos|:** {m['max_abs_cart_pos_m']:.2f} m &nbsp;&nbsp; "
        f"**Settle time after push:** {m['settle_time_after_disturbance_s']:.2f} s &nbsp;&nbsp; "
        f"**RMS force:** {m['rms_force_N']:.2f} N &nbsp;&nbsp; "
        f"**Fell over:** {'yes' if m['fell_over'] else 'no'}"
    )
    return gif_path, fig, metrics_md


with gr.Blocks(title="Cart-Pole: Classical Control vs RL") as demo:
    gr.Markdown(
        "# Cart-Pole: PID / LQR vs Reinforcement Learning\n"
        "Same nonlinear plant, three different controllers. "
        "LQR solves a Riccati equation on a linearized model; PID watches only the "
        "pole angle; the RL policy was trained from scratch by trial and error with PPO. "
        "Nudge the pole, add a disturbance push, and compare how each one recovers."
    )
    with gr.Row():
        with gr.Column(scale=1):
            controller_dd = gr.Dropdown(
                ["LQR (optimal, full state)", "PID (angle only)", "RL / PPO (learned)"],
                value="LQR (optimal, full state)",
                label="Controller",
            )
            theta_slider = gr.Slider(-20, 20, value=8.5, step=0.5, label="Initial pole tilt (degrees)")
            disturb_checkbox = gr.Checkbox(value=True, label="Apply a mid-run disturbance push")
            disturb_time_slider = gr.Slider(0.5, 7.0, value=3.0, step=0.1, label="Disturbance time (s)")
            disturb_strength_slider = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Disturbance strength (rad/s)")
            run_btn = gr.Button("Run simulation", variant="primary")
        with gr.Column(scale=1):
            gif_out = gr.Image(label="Animation", type="filepath")
    plot_out = gr.Plot(label="State and force traces")
    metrics_out = gr.Markdown()

    run_btn.click(
        run,
        inputs=[controller_dd, theta_slider, disturb_checkbox, disturb_time_slider, disturb_strength_slider],
        outputs=[gif_out, plot_out, metrics_out],
    )

if __name__ == "__main__":
    demo.launch()
