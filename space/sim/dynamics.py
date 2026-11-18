"""
Nonlinear cart-pole dynamics and a fourth-order Runge-Kutta integrator.

The cart-pole is a cart of mass M that can move along a frictionless
horizontal rail, carrying an inverted pendulum (a rigid rod of mass m and
half-length `length`, pivoting freely at the cart). A horizontal force F is
the only control input. State is x = [cart_pos, cart_vel, pole_angle,
pole_angvel], with pole_angle = 0 at the upright equilibrium.

Equations of motion (Florian 2007 / Barto-Sutton-Anderson 1983 form, derived
from Lagrangian mechanics for a uniform rod on a cart):

    temp      = (F + m*l*theta_dot^2*sin(theta)) / (M + m)
    theta_ddot = (g*sin(theta) - cos(theta)*temp)
                 / (l*(4/3 - m*cos(theta)^2/(M + m)))
    x_ddot    = temp - m*l*theta_ddot*cos(theta)/(M + m)

where l is the distance from the pivot to the pole's center of mass
(half the pole's physical length), and the 4/3 factor comes from the
rod's moment of inertia about its center, I = (1/3) m (2l)^2 = (4/3) m l^2,
via the parallel-axis theorem.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartPoleParams:
    gravity: float = 9.8
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_half_length: float = 0.5
    force_limit: float = 20.0
    track_limit: float = 2.4


def derivatives(state: np.ndarray, force: float, p: CartPoleParams) -> np.ndarray:
    _, x_dot, theta, theta_dot = state
    total_mass = p.cart_mass + p.pole_mass
    pole_mass_length = p.pole_mass * p.pole_half_length

    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    temp = (force + pole_mass_length * theta_dot**2 * sintheta) / total_mass
    theta_ddot = (p.gravity * sintheta - costheta * temp) / (
        p.pole_half_length * (4.0 / 3.0 - p.pole_mass * costheta**2 / total_mass)
    )
    x_ddot = temp - pole_mass_length * theta_ddot * costheta / total_mass

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


def rk4_step(state: np.ndarray, force: float, dt: float, p: CartPoleParams) -> np.ndarray:
    k1 = derivatives(state, force, p)
    k2 = derivatives(state + 0.5 * dt * k1, force, p)
    k3 = derivatives(state + 0.5 * dt * k2, force, p)
    k4 = derivatives(state + dt * k3, force, p)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def linearize(p: CartPoleParams, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Numeric Jacobian (central differences) of the dynamics about the
    upright equilibrium state=[0,0,0,0], force=0, giving the state-space
    matrices A (4x4) and B (4x1) for xdot = A x + B u used by the LQR design.
    """
    x0 = np.zeros(4)
    u0 = 0.0

    A = np.zeros((4, 4))
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        f_plus = derivatives(x0 + dx, u0, p)
        f_minus = derivatives(x0 - dx, u0, p)
        A[:, i] = (f_plus - f_minus) / (2 * eps)

    f_plus = derivatives(x0, u0 + eps, p)
    f_minus = derivatives(x0, u0 - eps, p)
    B = ((f_plus - f_minus) / (2 * eps)).reshape(4, 1)

    return A, B
