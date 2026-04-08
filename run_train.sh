#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

## PPO 
# uv run python -m mjlab.scripts.train Mjlab-Velocity-Flat-Unitree-G1 \
#   --agent.logger wandb \
#   --env.scene.num-envs 4096 \
#   --agent.num-steps-per-env 24 \
#   --agent.max-iterations 5000 \
#   --agent.run-name g1_ppo_paper_like

## FPO
uv run python -m fpo_mj.scripts.train Mjlab-Velocity-Flat-Unitree-G1 \
  --agent.logger wandb \
  --env.scene.num-envs 4096 \
  --agent.num-steps-per-env 24 \
  --agent.max-iterations 5000 \
  --agent.empirical-normalization \
  --agent.clip-actions 2.0 \
  --agent.policy.sampling-steps 64 \
  --agent.algorithm.n-samples-per-action 32 \
  --agent.algorithm.num-learning-epochs 32 \
  --agent.algorithm.num-mini-batches 4 \
  --agent.algorithm.learning-rate 1e-4 \
  --agent.algorithm.weight-decay 1e-4 \
  --agent.algorithm.clip-param 0.05 \
  --agent.algorithm.gamma 0.99 \
  --agent.algorithm.lam 0.95 \
  --agent.algorithm.ema-warmup-steps 500 \
  --agent.run-name g1_fpo_paper_like_norm_clip2_ema500

