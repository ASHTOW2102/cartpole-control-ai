---
title: Cart-Pole Control vs RL
emoji: 🎯
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.41.1
app_file: app.py
pinned: false
---

# Cart-Pole: Classical Control vs Reinforcement Learning

Pick a controller — **PID** (angle-only), **LQR** (optimal, full-state), or
a **PPO reinforcement-learning policy trained from scratch** — set an
initial pole tilt and an optional mid-run disturbance push, and watch the
same nonlinear cart-pole plant get balanced, with a live animation, state
and force traces, and the real recovery metrics (settle time, max tilt,
control effort).

Full write-up, equations of motion, and training code:
https://github.com/ASHTOW2102/cartpole-control-ai
