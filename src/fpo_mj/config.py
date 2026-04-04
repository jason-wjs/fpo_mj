"""Configuration dataclasses for fpo_mj."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl.config import RslRlBaseRunnerCfg


@dataclass
class FpoPolicyCfg:
  actor_hidden_dims: tuple[int, ...] = (256, 256, 256)
  critic_hidden_dims: tuple[int, ...] = (768, 768, 768)
  activation: str = "elu"
  compile_flow: bool = True
  actor_scale: float = 1.0
  actor_mlp_output_scale: float = 1.0
  actor_final_layer_weight_scale: float | None = None
  timestep_embed_dim: int = 8
  training_sampling_steps: int | None = None
  cfm_loss_t_inverse_cdf_beta: float = 1.0
  sampling_steps: int = 64
  cfm_loss_reduction: Literal["mean", "sum", "sqrt"] = "sqrt"
  action_perturb_std: float = 0.02


@dataclass
class FpoAlgorithmCfg:
  num_learning_epochs: int = 32
  num_mini_batches: int = 4
  learning_rate: float = 1e-4
  weight_decay: float = 1e-4
  adam_betas: tuple[float, float] = (0.9, 0.999)
  schedule: Literal["adaptive", "fixed"] = "fixed"
  gamma: float = 0.99
  lam: float = 0.95
  knn_entropy_coef: float = 0.0
  knn_entropy_k: int = 1
  desired_kl: float = 1e-4
  max_grad_norm: float = 1.0
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = False
  clip_param: float = 0.05
  trust_region_mode: Literal["ppo", "spo", "aspo"] = "aspo"
  normalize_advantage: bool = True
  normalize_advantage_per_mini_batch: bool = False
  advantage_clamp: tuple[float, float] = (100.0, 100.0)
  n_samples_per_action: int = 32
  cfm_diff_clamp_max: float = 10.0
  cfm_loss_clamp: float = 20.0
  cfm_loss_clamp_negative_advantages: bool = True
  cfm_loss_clamp_negative_advantages_max: float = 20.0
  storage_action_noise_std: float = 0.0
  ema_decay: float = 0.95
  ema_warmup_steps: int = 500


@dataclass
class FpoRunnerCfg(RslRlBaseRunnerCfg):
  """Runner config that preserves mjlab public runner fields and adds FPO knobs."""

  device: str = "cpu"
  empirical_normalization: bool = True
  randomize_reset_episode_progress: float = 0.0
  eval_episodes: int = 10
  flow_eval_modes: tuple[str, ...] = ("zero", "random")
  flow_eval_fixed_seed: int = 12345
  enable_post_training_eval: bool = True
  post_eval_checkpoint_interval: int = 1
  policy: FpoPolicyCfg = field(default_factory=FpoPolicyCfg)
  algorithm: FpoAlgorithmCfg = field(default_factory=FpoAlgorithmCfg)
