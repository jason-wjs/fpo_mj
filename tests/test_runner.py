from pathlib import Path

import torch
from tensordict import TensorDict

from fpo_mj.config import build_default_fpo_runner_cfg
from fpo_mj.runners import FpoOnPolicyRunner


class FakeVecEnv:
    def __init__(self):
        self.num_envs = 2
        self.num_actions = 3
        self.device = torch.device("cpu")
        self.max_episode_length = 4
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.common_step_counter = 0
        self.unwrapped = self
        self._step_count = 0

    def get_observations(self):
        return TensorDict(
            {
                "actor": torch.randn(self.num_envs, 5),
                "critic": torch.randn(self.num_envs, 7),
            },
            batch_size=[self.num_envs],
        )

    def reset(self):
        self._step_count = 0
        self.episode_length_buf.zero_()
        return self.get_observations(), {"log": {}}

    def step(self, actions):
        assert actions.shape == (self.num_envs, self.num_actions)
        self._step_count += 1
        self.common_step_counter += self.num_envs
        self.episode_length_buf += 1
        dones = torch.ones(self.num_envs, dtype=torch.long) if self._step_count >= 2 else torch.zeros(self.num_envs, dtype=torch.long)
        if dones.any():
            self.episode_length_buf[dones > 0] = 0
        return self.get_observations(), torch.ones(self.num_envs), dones, {"log": {}}

    def close(self):
        return None


def _make_cfg():
    cfg = build_default_fpo_runner_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    cfg.logger = "tensorboard"
    cfg.num_steps_per_env = 2
    cfg.max_iterations = 1
    cfg.save_interval = 1
    cfg.policy.actor_hidden_dims = (8, 8)
    cfg.policy.critic_hidden_dims = (16, 16)
    cfg.policy.timestep_embed_dim = 4
    cfg.policy.sampling_steps = 4
    cfg.algorithm.num_learning_epochs = 1
    cfg.algorithm.num_mini_batches = 1
    cfg.algorithm.n_samples_per_action = 4
    return cfg


def test_runner_learn_writes_checkpoint(tmp_path):
    runner = FpoOnPolicyRunner(FakeVecEnv(), _make_cfg(), log_dir=str(tmp_path), device="cpu")

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    checkpoints = list(Path(tmp_path).glob("model_*.pt"))
    assert checkpoints


def test_runner_load_restores_iteration_and_env_state(tmp_path):
    runner = FpoOnPolicyRunner(FakeVecEnv(), _make_cfg(), log_dir=str(tmp_path), device="cpu")
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    checkpoint_path = sorted(Path(tmp_path).glob("model_*.pt"))[-1]

    new_env = FakeVecEnv()
    new_runner = FpoOnPolicyRunner(new_env, _make_cfg(), log_dir=str(tmp_path / "reload"), device="cpu")
    infos = new_runner.load(str(checkpoint_path))

    assert new_runner.current_learning_iteration == 0
    assert infos["env_state"]["common_step_counter"] == new_env.common_step_counter


def test_runner_evaluate_returns_metrics_for_each_mode(tmp_path):
    runner = FpoOnPolicyRunner(FakeVecEnv(), _make_cfg(), log_dir=str(tmp_path), device="cpu")

    results = runner.evaluate(num_episodes=2, eval_modes=("zero", "random"))

    assert set(results) == {"zero", "random"}
    for mode in results.values():
        assert mode["num_episodes"] >= 2
        assert "mean_reward" in mode
        assert "mean_episode_length" in mode
