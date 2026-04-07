from fpo_mj.config import FpoRunnerCfg, build_default_fpo_runner_cfg
from fpo_mj.supported_tasks import SUPPORTED_TASKS


def test_supported_tasks_contains_g1_velocity() -> None:
    assert SUPPORTED_TASKS == ("Mjlab-Velocity-Flat-Unitree-G1",)


def test_default_runner_cfg_builds_for_each_supported_task() -> None:
    built = [build_default_fpo_runner_cfg(task_id) for task_id in SUPPORTED_TASKS]

    assert len(built) == len(SUPPORTED_TASKS)
    assert all(isinstance(cfg, FpoRunnerCfg) for cfg in built)
