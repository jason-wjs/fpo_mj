#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

## FPO
uv run python -m fpo_mj.scripts.play Mjlab-Velocity-Flat-Unitree-G1 \
  --agent-type fpo \
  --checkpoint-path /home/humanoid/Projects/Junsong_WU/learning/locomotion/fpo_mj/logs/fpo_mj/g1_flat_fpo/2026-04-05_12-34-31_g1_fpo_paper_like/model_4999.pt \
  --viewer viser



## PPO
# uv run python -m fpo_mj.scripts.play Mjlab-Velocity-Flat-Unitree-G1 \
#   --agent-type ppo \
#   --checkpoint-path /home/humanoid/Projects/Junsong_WU/learning/locomotion/fpo_mj/logs/rsl_rl/g1_velocity/2026-04-05_00-35-17_g1_ppo_paper_like/model_4999.pt \
#   --viewer viser
