import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from fpo_mj.config import build_default_fpo_runner_cfg
from fpo_mj.runners import FpoOnPolicyRunner


def test_real_mjlab_g1_smoke_train(tmp_path):
    env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    env_cfg.scene.num_envs = 2

    agent_cfg = build_default_fpo_runner_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    agent_cfg.logger = "tensorboard"
    agent_cfg.num_steps_per_env = 2
    agent_cfg.max_iterations = 1
    agent_cfg.save_interval = 1
    agent_cfg.policy.actor_hidden_dims = (16, 16)
    agent_cfg.policy.critic_hidden_dims = (16, 16)
    agent_cfg.policy.timestep_embed_dim = 4
    agent_cfg.policy.sampling_steps = 4
    agent_cfg.algorithm.num_learning_epochs = 1
    agent_cfg.algorithm.num_mini_batches = 1
    agent_cfg.algorithm.n_samples_per_action = 4

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = FpoOnPolicyRunner(env, agent_cfg, log_dir=str(tmp_path), device="cpu")

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    assert list(tmp_path.glob("model_*.pt"))
    env.close()
