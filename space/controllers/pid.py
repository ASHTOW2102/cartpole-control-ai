"""
PID controller for the cart-pole, stabilizing the pole angle only.

u(t) = Kp*e(t) + Ki*integral(e) dt + Kd*de/dt,   e(t) = theta(t) - 0

Note the sign: a cart-pole is stabilized by pushing the cart *toward* the
side the pole is leaning (the same way you'd slide your hand under a
falling broomstick), so the restoring force must carry the same sign as
the angle error, not the opposite. This matches the sign of the theta
term in the LQR gain K computed in controllers/lqr.py.

This is the classic "balance-only" baseline: it watches a single error
signal (pole angle) and ignores cart position entirely, so unlike LQR it
has no mechanism to recenter the cart and will drift along the track while
still holding the pole upright. That drift-vs-no-drift contrast against LQR
is itself one of the comparison's results.
"""
import numpy as np

from sim.dynamics import CartPoleParams


class PIDController:
    def __init__(self, params: CartPoleParams, kp=60.0, ki=5.0, kd=12.0):
        self.params = params
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def act(self, state: np.ndarray, dt: float = 0.02) -> float:
        theta = state[2]
        error = theta - 0.0
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        force = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(force, -self.params.force_limit, self.params.force_limit))
