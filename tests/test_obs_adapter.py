import pytest
import torch
from mjlab.tasks.registry import load_rl_cfg
from tensordict import TensorDict

from fpo_mj.env import ObservationAdapter


def test_observation_adapter_for_g1_preserves_flat_groups() -> None:
    obs_groups = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1").obs_groups
    assert obs_groups == {"actor": ("actor",), "critic": ("critic",)}
    adapter = ObservationAdapter(obs_groups)

    obs = TensorDict(
        {
            "actor": torch.randn(2, 99),
            "critic": torch.randn(2, 111),
        },
        batch_size=[2],
    )

    actor_obs, critic_obs = adapter.adapt(obs)

    assert actor_obs.shape == (2, 99)
    assert critic_obs.shape == (2, 111)
    assert torch.equal(actor_obs, obs["actor"])
    assert torch.equal(critic_obs, obs["critic"])


def test_observation_adapter_concatenates_multiple_keys_in_order() -> None:
    adapter = ObservationAdapter({"actor": ("proprio", "history"), "critic": ("proprio",)})
    obs = TensorDict(
        {
            "proprio": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "history": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        },
        batch_size=[2],
    )

    actor_obs, critic_obs = adapter.adapt(obs)

    expected = torch.cat((obs["proprio"], obs["history"]), dim=-1)
    assert actor_obs.shape == (2, 7)
    assert torch.equal(actor_obs, expected)
    assert torch.equal(critic_obs, obs["proprio"])


def test_observation_adapter_raises_for_missing_key() -> None:
    obs = TensorDict({"actor": torch.randn(2, 3)}, batch_size=[2])
    adapter = ObservationAdapter({"actor": ("actor",), "critic": ("critic",)})

    with pytest.raises(KeyError, match="critic"):
        adapter.adapt(obs)


def test_observation_adapter_raises_for_non_matrix_tensor() -> None:
    obs = TensorDict({"actor": torch.randn(2, 3, 4)}, batch_size=[2])
    adapter = ObservationAdapter({"actor": ("actor",), "critic": ("actor",)})

    with pytest.raises(ValueError, match="2D"):
        adapter.adapt(obs)
