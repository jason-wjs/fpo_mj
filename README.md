# fpo_mj

FPO++ training on top of official `mjlab` tasks.

This repository currently targets a single task:

- `Mjlab-Velocity-Flat-Unitree-G1`

The immediate goal is to run FPO on Unitree G1 locomotion with `mjlab==1.2.0` and compare it against the official PPO baseline under the same task and rollout protocol.

## Design Goals

- Use the official PyPI release `mjlab==1.2.0`
- Keep local `mjlab` source trees untouched
- Reuse `mjlab` task loading and runner-facing config semantics
- Add a standalone FPO runner, observation adapter, checkpointing, and evaluation flow
- Make PPO vs. FPO comparisons easy to reproduce from the terminal

## Scope

What is implemented in `v0.1.0`:

- FPO actor/critic, storage, normalizers, EMA, and on-policy runner
- `TensorDict` observation adaptation driven by `mjlab` `obs_groups`
- Training entrypoint for `Mjlab-Velocity-Flat-Unitree-G1`
- Checkpoint save/load and resume support
- Evaluation in both `zero` and `random` flow sampling modes
- Unit tests plus a real `mjlab` G1 smoke test

What is intentionally not in scope yet:

- Multi-task support beyond G1 locomotion
- Viewer or play scripts
- Export flows such as ONNX or TorchScript packaging
- Changes inside the `mjlab` repository

## Repository Layout

```text
src/fpo_mj/
  algorithms/     FPO update logic
  config/         Runner, policy, and algorithm dataclasses
  env/            TensorDict observation adapter
  modules/        Actor-critic, EMA, normalization
  runners/        FpoOnPolicyRunner
  scripts/        train.py, eval.py, and play.py
  storage/        rollout storage
  supported_tasks.py
  utils/          small shared helpers
tests/
  ...             unit tests and smoke integration tests
```

Stable public module entrypoints:

- `fpo_mj.config`
- `fpo_mj.env`
- `fpo_mj.runners`
- `fpo_mj.scripts.train`
- `fpo_mj.scripts.eval`
- `fpo_mj.scripts.play`

## Requirements

- Linux
- Python `>=3.10,<3.14`
- `uv`
- NVIDIA GPU recommended for real training

`mjlab` and the rest of the Python dependencies are installed from `pyproject.toml`.

## Setup

Create the project environment and install dev dependencies:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/fpo_mj
uv sync --extra dev
```

All commands below assume you run them from the repository root.

## Training

### Quick Smoke Run

Use this first to confirm the task, MuJoCo stack, and logging path all work:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m fpo_mj.scripts.train Mjlab-Velocity-Flat-Unitree-G1 \
  --agent.logger tensorboard \
  --env.scene.num-envs 64 \
  --agent.max-iterations 1 \
  --agent.run-name g1_fpo_smoke
```

### Paper-Like FPO Run

This is the closest command in this repository to the locomotion settings reported in the FPO++ paper for humanoids:

- `4096` environments
- `24` steps per environment between updates
- `2000` policy updates
- `64` flow sampling steps
- `32` samples per action
- `32` learning epochs
- `4` mini-batches
- `lr=1e-4`, `weight_decay=1e-4`, `clip=0.05`, `gamma=0.99`, `lambda=0.95`

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m fpo_mj.scripts.train Mjlab-Velocity-Flat-Unitree-G1 \
  --agent.logger tensorboard \
  --env.scene.num-envs 4096 \
  --agent.num-steps-per-env 24 \
  --agent.max-iterations 2000 \
  --agent.policy.sampling-steps 64 \
  --agent.algorithm.n-samples-per-action 32 \
  --agent.algorithm.num-learning-epochs 32 \
  --agent.algorithm.num-mini-batches 4 \
  --agent.algorithm.learning-rate 1e-4 \
  --agent.algorithm.weight-decay 1e-4 \
  --agent.algorithm.clip-param 0.05 \
  --agent.algorithm.gamma 0.99 \
  --agent.algorithm.lam 0.95 \
  --agent.run-name g1_fpo_paper_like
```

Important caveat: this matches the paper's reported hyperparameters as closely as possible, but it is still running on `mjlab`, not on the original IsaacLab locomotion environment used in the paper.

### Practical Single-GPU Run

For an RTX 4090, `1024` or `2048` environments are often a safer starting point for a long run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m fpo_mj.scripts.train Mjlab-Velocity-Flat-Unitree-G1 \
  --agent.logger tensorboard \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 30000 \
  --agent.run-name g1_fpo_4090
```

## PPO Baseline

To compare against the official `mjlab` PPO pipeline on the same task:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m mjlab.scripts.train Mjlab-Velocity-Flat-Unitree-G1 \
  --agent.logger tensorboard \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 30000 \
  --agent.run-name g1_ppo_4090
```

For a fair comparison:

- keep the same `task_id`
- keep the same seed set
- keep the same `num_envs`
- keep the same `num_steps_per_env`
- compare at the same total environment step budget

## Evaluation

After training, evaluate a checkpoint with both `zero` and `random` flow sampling modes:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m fpo_mj.scripts.eval Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint-path logs/fpo_mj/g1_flat_fpo/<run_dir>/model_1999.pt \
  --num-episodes 10 \
  --eval-modes zero random
```

You can also resolve a checkpoint by run directory instead of passing a full path:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m fpo_mj.scripts.eval Mjlab-Velocity-Flat-Unitree-G1 \
  --agent.load-run <run_dir> \
  --agent.load-checkpoint model_1999.pt \
  --num-episodes 10 \
  --eval-modes zero random
```

## Logs And Artifacts

FPO logs are written under:

```text
logs/fpo_mj/g1_flat_fpo/<timestamp>_<run_name>/
```

Typical contents:

- `params/env.yaml`
- `params/agent.yaml`
- `events.out.tfevents...`
- `model_<iteration>.pt`

## Testing

Run the full test suite:

```bash
PYTHONPATH=src uv run pytest -q
```

Run only the real `mjlab` G1 smoke test:

```bash
PYTHONPATH=src uv run pytest -q tests/test_integration_smoke.py -s
```

## Notes On The Paper Mapping

The FPO++ paper reports locomotion hyperparameters in the paper body and Appendix A. In this repository:

- the paper-like launch command mirrors those reported values
- task loading comes from `mjlab`
- the current supported task is `Mjlab-Velocity-Flat-Unitree-G1`

This means the repository is suitable for controlled PPO vs. FPO comparisons on `mjlab`, but it is not an exact IsaacLab reproduction package.

## Play / Visualization

Use `play.py` for qualitative inspection of trained checkpoints. Unlike `eval.py`, this command opens a viewer and plays a policy in real time instead of aggregating episode metrics.

### FPO Playback

FPO playback supports flow initialization modes:

- `zero`: start the flow from zeros
- `random`: start the flow from fresh Gaussian noise
- `fixed_seed`: start from deterministic Gaussian noise using `eval_fixed_seed`

Example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m fpo_mj.scripts.play Mjlab-Velocity-Flat-Unitree-G1 \
  --agent-type fpo \
  --checkpoint-path logs/fpo_mj/g1_flat_fpo/<run_dir>/model_4999.pt \
  --eval-mode zero \
  --viewer native
```

### PPO Playback

The same command also supports official `mjlab` PPO checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m fpo_mj.scripts.play Mjlab-Velocity-Flat-Unitree-G1 \
  --agent-type ppo \
  --checkpoint-path logs/rsl_rl/g1_velocity/<run_dir>/model_4999.pt \
  --viewer native
```

### Quick Launcher

The repository root includes [`run_play.sh`](run_play.sh), matching the style of `run_train.sh`. It defaults to an FPO playback command and keeps PPO and alternate FPO modes as commented templates for quick editing.

## License

This project is released under the BSD 3-Clause License. See [LICENSE](LICENSE).

Third-party notice details are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
