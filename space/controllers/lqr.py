"""
LQR (Linear-Quadratic Regulator) controller for the cart-pole.

The nonlinear plant is linearized about the upright equilibrium (see
sim.dynamics.linearize), giving xdot = A x + B u. LQR picks the gain K that
minimizes the infinite-horizon cost

    J = integral_0^inf ( x^T Q x + u^T R u ) dt

by solving the continuous-time algebraic Riccati equation

    A^T P + P A - P B R^-1 B^T P + Q = 0

and setting K = R^-1 B^T P, with optimal control u = -K x (x measured as a
deviation from the upright, centered equilibrium).
"""
import numpy as np
from control import lqr

from sim.dynamics import CartPoleParams, linearize


class LQRController:
    def __init__(self, params: CartPoleParams, Q=None, R=None):
        self.params = params
        A, B = linearize(params)
        self.A = A
        self.B = B
        self.Q = np.diag([1.0, 1.0, 10.0, 1.0]) if Q is None else Q
        self.R = np.array([[0.1]]) if R is None else R
        K, _, _ = lqr(A, B, self.Q, self.R)
        self.K = np.asarray(K)

    def act(self, state: np.ndarray) -> float:
        force = float(-self.K @ state)
        return float(np.clip(force, -self.params.force_limit, self.params.force_limit))

    def reset(self):
        pass
