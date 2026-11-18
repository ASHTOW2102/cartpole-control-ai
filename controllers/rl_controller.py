"""Wraps a trained stable-baselines3 PPO policy as a controller with the
same act(state) -> force interface as LQRController and PIDController.

The policy was trained on observations normalized by a running mean/variance
(see rl/train.py's VecNormalize). Those statistics are frozen artifacts of
training, not something to refit here -- at inference time we load them and
apply the exact same transform to each raw state before querying the policy,
otherwise the policy sees out-of-distribution inputs and acts erratically.
"""
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from sim.dynamics import CartPoleParams


class RLController:
    def __init__(self, model_path: str, params: CartPoleParams, vecnorm_path: str | None = None):
        self.model = PPO.load(model_path)
        self.params = params

        if vecnorm_path is None:
            candidate = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
            vecnorm_path = candidate if os.path.exists(candidate) else None

        self._obs_rms = None
        self._clip_obs = 10.0
        if vecnorm_path is not None:
            from rl.env import CartPoleContinuousEnv

            dummy_venv = make_vec_env(lambda: CartPoleContinuousEnv(), n_envs=1)
            vec_normalize = VecNormalize.load(vecnorm_path, dummy_venv)
            self._obs_rms = vec_normalize.obs_rms
            self._clip_obs = vec_normalize.clip_obs

    def reset(self):
        pass

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        if self._obs_rms is None:
            return state
        normalized = (state - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + 1e-8)
        return np.clip(normalized, -self._clip_obs, self._clip_obs)

    def act(self, state: np.ndarray) -> float:
        obs = self._normalize(state.astype(np.float32))
        action, _ = self.model.predict(obs, deterministic=True)
        force = float(action[0])
        return float(np.clip(force, -self.params.force_limit, self.params.force_limit))
