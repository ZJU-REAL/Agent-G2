from __future__ import annotations

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = StepHintTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class StepHintTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        from stephint.patch import apply_stephint_patches

        apply_stephint_patches()
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.trainer.main_ppo import build_ppo_trainer

        trainer = build_ppo_trainer(config, local_path=local_path)
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
