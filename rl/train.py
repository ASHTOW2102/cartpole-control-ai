"""
Trains a PPO agent (stable-baselines3) on the continuous cart-pole
environment and saves the policy to rl/ppo_cartpole.zip.

The observation components live on very different scales (pole angle is
a fraction of a radian, cart position is up to a couple of meters), which
starves a vanilla MLP policy of a useful gradient signal. VecNormalize
tracks running mean/variance of observations and rewards and rescales
both, which is the standard SB3 fix for exactly this failure mode -- its
statistics are saved alongside the policy and must be reapplied
(unchanged, not re-fit) at inference time; see controllers/rl_controller.py.

Run from the project root:  python -m rl.train
"""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from rl.env import CartPoleContinuousEnv

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ppo_cartpole.zip")
VECNORM_PATH = os.path.join(os.path.dirname(__file__), "vecnormalize.pkl")


def main(total_timesteps: int = 700_000):
    env = make_vec_env(lambda: Monitor(CartPoleContinuousEnv()), n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        gamma=0.98,
        verbose=1,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)
    print(f"Saved trained policy to {MODEL_PATH}")
    print(f"Saved observation/reward normalization stats to {VECNORM_PATH}")


if __name__ == "__main__":
    main()
