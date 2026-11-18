"""Shared simulation-and-metrics logic used by both compare.py (local,
batch comparison across all three controllers) and the Hugging Face
Gradio app (interactive, single-controller runs)."""
import numpy as np

from sim.dynamics import CartPoleParams, rk4_step


def simulate(
    controller,
    params: CartPoleParams,
    initial_theta: float = 0.15,
    disturbance_time: float | None = 3.0,
    disturbance_theta_dot: float = 2.0,
    duration_s: float = 8.0,
    dt: float = 0.02,
):
    n_steps = int(duration_s / dt)
    times = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, 4))
    forces = np.zeros(n_steps)

    controller.reset()
    state = np.array([0.0, 0.0, initial_theta, 0.0])
    states[0] = state

    for i in range(n_steps):
        t = i * dt
        if disturbance_time is not None and abs(t - disturbance_time) < dt / 2:
            state[3] += disturbance_theta_dot

        try:
            force = controller.act(state, dt)
        except TypeError:
            force = controller.act(state)

        state = rk4_step(state, force, dt, params)
        times[i + 1] = t + dt
        states[i + 1] = state
        forces[i] = force

        if abs(state[2]) > np.pi / 2:
            states[i + 2 :] = state
            times = times[: i + 2]
            states = states[: i + 2]
            forces = forces[: i + 1]
            break

    return times, states, forces


def compute_metrics(times, states, forces, name, disturbance_time=3.0, settle_threshold=0.03):
    theta = states[:, 2]
    x = states[:, 0]
    after = times >= disturbance_time
    above = np.abs(theta[after]) >= settle_threshold
    above_idx = np.where(above)[0]
    if len(above_idx) > 0:
        # time of the LAST moment theta is still outside the settle band --
        # i.e. how long the disturbance's swing takes to fully die out,
        # not just the first (trivial, pre-swing) instant it happens to dip
        # under the threshold.
        settle_time = times[after][above_idx[-1]] - disturbance_time
    else:
        settle_time = 0.0
    return {
        "controller": name,
        "max_abs_theta_deg": float(np.degrees(np.max(np.abs(theta)))),
        "max_abs_cart_pos_m": float(np.max(np.abs(x))),
        "settle_time_after_disturbance_s": float(settle_time),
        "rms_force_N": float(np.sqrt(np.mean(np.square(forces)))) if len(forces) else 0.0,
        "fell_over": bool(np.max(np.abs(theta)) > np.pi / 2),
    }
