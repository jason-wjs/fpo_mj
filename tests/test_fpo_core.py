import pytest
import torch

from fpo_mj.algorithms import FPO
from fpo_mj.config import FpoAlgorithmCfg, FpoPolicyCfg
from fpo_mj.modules import ActorCritic, EmpiricalNormalization
from fpo_mj.modules.ema import ExponentialMovingAverage
from fpo_mj.storage import RolloutStorage
from fpo_mj.utils import resolve_nn_activation


def test_resolve_nn_activation_rejects_unknown_name() -> None:
    assert isinstance(resolve_nn_activation("elu"), torch.nn.ELU)
    with pytest.raises(ValueError, match="Invalid activation function"):
        resolve_nn_activation("does-not-exist")


def test_empirical_normalization_updates_running_stats() -> None:
    norm = EmpiricalNormalization(shape=(2,))
    batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    normalized = norm(batch)

    assert normalized.shape == batch.shape
    assert torch.allclose(norm.mean, torch.tensor([2.0, 3.0]))
    assert torch.all(norm.std > 0)


def test_exponential_moving_average_tracks_parameter_updates() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))

    ema = ExponentialMovingAverage(model, decay=0.5, device=torch.device("cpu"))
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 5.0]]))
    ema.update()

    assert torch.allclose(ema.shadow_params["weight"], torch.tensor([[2.0, 3.5]]))


def test_rollout_storage_computes_returns_and_batches() -> None:
    storage = RolloutStorage(
        num_envs=2,
        num_transitions_per_env=2,
        obs_shape=[4],
        privileged_obs_shape=[6],
        actions_shape=[2],
        device="cpu",
        n_samples_per_action=3,
    )

    for step in range(2):
        transition = RolloutStorage.Transition()
        transition.observations = torch.full((2, 4), float(step))
        transition.privileged_observations = torch.full((2, 6), float(step))
        transition.actions = torch.zeros(2, 2)
        transition.rewards = torch.ones(2)
        transition.dones = torch.tensor([0, 1] if step == 1 else [0, 0])
        transition.values = torch.zeros(2, 1)
        transition.initial_cfm_loss = torch.zeros(2, 3)
        transition.cfm_loss_eps = torch.zeros(2, 3, 2)
        transition.cfm_loss_t = torch.zeros(2, 3, 1)
        transition.x1_pred = torch.zeros(2, 3, 2)
        storage.add_transitions(transition)

    storage.compute_returns(
        last_values=torch.zeros(2, 1),
        gamma=0.99,
        lam=0.95,
        normalize_advantage=True,
    )

    batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    obs_batch, critic_obs_batch, actions_batch = batch[:3]
    assert obs_batch.shape == (4, 4)
    assert critic_obs_batch.shape == (4, 6)
    assert actions_batch.shape == (4, 2)


def test_actor_critic_action_and_cfm_shapes() -> None:
    policy = ActorCritic(
        num_actor_obs=4,
        num_critic_obs=6,
        num_actions=2,
        cfg=FpoPolicyCfg(
          actor_hidden_dims=(16, 16),
          critic_hidden_dims=(16, 16),
          sampling_steps=4,
        ),
    )
    obs = torch.randn(3, 4)
    critic_obs = torch.randn(3, 6)
    actions = policy.act(obs)
    values = policy.evaluate(critic_obs)
    eps = torch.randn(3, 2, 2)
    t = torch.rand(3, 2, 1)
    cfm_loss, x1_pred, x0_pred = policy.get_cfm_loss(obs, actions, eps, t)

    assert actions.shape == (3, 2)
    assert values.shape == (3, 1)
    assert cfm_loss.shape == (3, 2)
    assert x1_pred.shape == (3, 2, 2)
    assert x0_pred.shape == (3, 2, 2)


def test_fpo_collects_rollout_and_updates() -> None:
    policy = ActorCritic(
        num_actor_obs=4,
        num_critic_obs=6,
        num_actions=2,
        cfg=FpoPolicyCfg(
          actor_hidden_dims=(16, 16),
          critic_hidden_dims=(16, 16),
          sampling_steps=4,
        ),
    )
    algorithm = FPO(
        policy,
        cfg=FpoAlgorithmCfg(
          num_learning_epochs=1,
          num_mini_batches=1,
          n_samples_per_action=2,
          ema_decay=0.0,
        ),
        device="cpu",
    )
    algorithm.init_storage(
        num_envs=2,
        num_transitions_per_env=2,
        actor_obs_shape=[4],
        critic_obs_shape=[6],
        actions_shape=[2],
    )

    for step in range(2):
        obs = torch.randn(2, 4)
        critic_obs = torch.randn(2, 6)
        algorithm.act(obs, critic_obs)
        rewards = torch.ones(2)
        dones = torch.tensor([0, 1] if step == 1 else [0, 0])
        algorithm.process_env_step(rewards, dones, infos={})

    algorithm.compute_returns(torch.randn(2, 6))
    loss_dict = algorithm.update()

    assert "surrogate_loss" in loss_dict
    assert "value_loss" in loss_dict
    assert "metrics" in loss_dict
