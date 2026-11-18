"""
Continuous-action Gymnasium environment wrapping the same nonlinear
cart-pole dynamics used by the LQR and PID controllers, so all three
controllers are compared on an identical plant.

Action: a single continuous force in [-force_limit, force_limit] (N).
Observation: [cart_pos, cart_vel, pole_angle, pole_angvel].
Reward: -(theta^2 + 0.1*x^2 + 0.001*force^2) per step, i.e. the agent is
penalized for tilting, for drifting off-center, and lightly for control
effort -- an LQR-like quadratic cost, so the RL agent is optimizing a
similar objective to LQR but discovering its own policy from experience
rather than from a linearized model. A large one-off FALL_PENALTY is added
on the step that ends the episode by falling: the ordinary per-step cost
is tiny (bounded by ~0.25 for theta alone, since episodes end at 0.5 rad)
and on its own gives almost no incentive to avoid the failure state versus
just drifting near it, so without an explicit penalty PPO has little reason
to spend control effort preventing a fall.
Episode ends if the pole falls past +-0.5 rad or the cart leaves the track.

Each episode also schedules exactly one random disturbance push (a jump in
pole angular velocity, applied at a random step) so the trained policy has
actually experienced being knocked off balance, not just small random
initial tilts. Without this, a policy that only ever sees gentle starts
over-fits to the training distribution and faceplants the moment a
disturbance pushes it somewhere it has never been.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.dynamics import CartPoleParams, rk4_step

ANGLE_LIMIT = 0.5
MAX_STEPS = 500
DT = 0.02
FALL_PENALTY = 50.0


class CartPoleContinuousEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, params: CartPoleParams | None = None):
        super().__init__()
        self.params = params or CartPoleParams()
        self.action_space = spaces.Box(
            low=-self.params.force_limit, high=self.params.force_limit, shape=(1,), dtype=np.float32
        )
        high = np.array([self.params.track_limit * 2, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.state = np.zeros(4)
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        theta0 = self.np_random.uniform(-0.15, 0.15)
        x0 = self.np_random.uniform(-0.3, 0.3)
        self.state = np.array([x0, 0.0, theta0, 0.0])
        self._steps = 0
        self._disturbance_step = int(self.np_random.integers(50, MAX_STEPS - 50))
        self._disturbance_strength = float(self.np_random.uniform(-3.0, 3.0))
        return self.state.astype(np.float32), {}

    def step(self, action):
        force = float(np.clip(action[0], -self.params.force_limit, self.params.force_limit))
        self.state = rk4_step(self.state, force, DT, self.params)
        self._steps += 1

        if self._steps == self._disturbance_step:
            self.state[3] += self._disturbance_strength

        x, _, theta, _ = self.state
        reward = -(theta**2 + 0.1 * x**2 + 0.001 * force**2)

        terminated = bool(abs(theta) > ANGLE_LIMIT or abs(x) > self.params.track_limit)
        if terminated:
            reward -= FALL_PENALTY
        truncated = self._steps >= MAX_STEPS
        return self.state.astype(np.float32), float(reward), terminated, truncated, {}
