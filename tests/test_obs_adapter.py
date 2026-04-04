import pytest
import torch
from mjlab.tasks.registry import load_rl_cfg
from tensordict import TensorDict

from fpo_mj.obs_adapter import (
    extract_actor_critic_observations,
    extract_group_observation,
)


def test_extract_actor_critic_observations_for_g1_preserves_flat_groups() -> None:
    obs_groups = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1").obs_groups
    assert obs_groups == {"actor": ("actor",), "critic": ("critic",)}

    obs = TensorDict(
        {
            "actor": torch.randn(2, 99),
            "critic": torch.randn(2, 111),
        },
        batch_size=[2],
    )

    actor_obs, critic_obs = extract_actor_critic_observations(obs, obs_groups)

    assert actor_obs.shape == (2, 99)
    assert critic_obs.shape == (2, 111)
    assert torch.equal(actor_obs, obs["actor"])
    assert torch.equal(critic_obs, obs["critic"])


def test_extract_group_observation_concatenates_multiple_keys_in_order() -> None:
    obs = TensorDict(
        {
            "proprio": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "history": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        },
        batch_size=[2],
    )

    actor_obs = extract_group_observation(
        obs,
        group_name="actor",
        obs_groups={"actor": ("proprio", "history")},
    )

    expected = torch.cat((obs["proprio"], obs["history"]), dim=-1)
    assert actor_obs.shape == (2, 7)
    assert torch.equal(actor_obs, expected)


def test_extract_group_observation_raises_for_missing_key() -> None:
    obs = TensorDict({"actor": torch.randn(2, 3)}, batch_size=[2])

    with pytest.raises(KeyError, match="critic"):
        extract_group_observation(
            obs,
            group_name="critic",
            obs_groups={"critic": ("critic",)},
        )


def test_extract_group_observation_raises_for_non_matrix_tensor() -> None:
    obs = TensorDict({"actor": torch.randn(2, 3, 4)}, batch_size=[2])

    with pytest.raises(ValueError, match="2D"):
        extract_group_observation(
            obs,
            group_name="actor",
            obs_groups={"actor": ("actor",)},
        )
