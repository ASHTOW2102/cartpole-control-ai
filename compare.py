"""
Runs PID, LQR, and a trained RL (PPO) controller on the identical nonlinear
cart-pole plant, from the identical initial condition, with the identical
mid-simulation disturbance push -- then reports recovery metrics, saves a
comparison plot, and renders an animated GIF per controller.

Run from the project root:  python compare.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controllers.lqr import LQRController
from controllers.pid import PIDController
from controllers.rl_controller import RLController
from sim.animate import render_animation
from sim.dynamics import CartPoleParams
from sim.experiment import compute_metrics, simulate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DISTURBANCE_TIME = 3.0


def main():
    params = CartPoleParams()

    controllers = {
        "PID": PIDController(params),
        "LQR": LQRController(params),
    }
    rl_path = os.path.join(os.path.dirname(__file__), "rl", "ppo_cartpole.zip")
    if os.path.exists(rl_path):
        controllers["RL (PPO)"] = RLController(rl_path, params)
    else:
        print("No trained RL model found at rl/ppo_cartpole.zip -- skipping RL. "
              "Run `python -m rl.train` first.")

    runs = {}
    all_metrics = []
    for name, controller in controllers.items():
        times, states, forces = simulate(controller, params, disturbance_time=DISTURBANCE_TIME)
        runs[name] = (times, states, forces)
        m = compute_metrics(times, states, forces, name, disturbance_time=DISTURBANCE_TIME)
        all_metrics.append(m)
        print(m)

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        render_animation(
            times, states, params,
            title=f"Cart-Pole balancing -- {name}",
            save_path=os.path.join(RESULTS_DIR, f"animation_{safe_name}.gif"),
        )

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for name, (times, states, forces) in runs.items():
        axes[0].plot(times, np.degrees(states[:, 2]), label=name)
        axes[1].plot(times, states[:, 0], label=name)
        axes[2].plot(times[:-1], forces, label=name)

    axes[0].axvline(DISTURBANCE_TIME, color="gray", linestyle="--", linewidth=1, label="disturbance")
    axes[0].set_ylabel("pole angle (deg)")
    axes[1].set_ylabel("cart position (m)")
    axes[2].set_ylabel("force (N)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("PID vs LQR vs RL -- cart-pole disturbance recovery")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "comparison_plot.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "metrics.md"), "w") as f:
        f.write("| Controller | Max |theta| (deg) | Max |x| (m) | Settle time after push (s) | RMS force (N) | Fell over |\n")
        f.write("|---|---|---|---|---|---|\n")
        for m in all_metrics:
            f.write(
                f"| {m['controller']} | {m['max_abs_theta_deg']:.2f} | {m['max_abs_cart_pos_m']:.2f} | "
                f"{m['settle_time_after_disturbance_s']:.2f} | {m['rms_force_N']:.2f} | {m['fell_over']} |\n"
            )

    print(f"\nSaved comparison plot, metrics table, and per-controller animations to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
