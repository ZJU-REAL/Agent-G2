from __future__ import annotations

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_webshop_sft_only(config)


def run_webshop_sft_only(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = WebshopSFTOnlyTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class WebshopSFTOnlyTaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trainer = build_webshop_sft_only_trainer(config, local_path=local_path)
        trainer.init_workers()
        trainer.fit()


def build_webshop_sft_only_trainer(config, local_path=None, build_envs=True):
    if local_path is None:
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

    if build_envs:
        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)
    else:
        envs, val_envs = None, None

    from verl.utils import hf_processor, hf_tokenizer

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    if config.actor_rollout_ref.rollout.name in ["vllm"]:
        from verl.utils.vllm_utils import is_version_ge

        if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
            if not is_version_ge(pkg="vllm", minver="0.7.3"):
                raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    if int(config.actor_rollout_ref.rollout.n) != 1:
        raise ValueError("actor_rollout_ref.rollout.n must be 1 for env-based SFT-only training.")

    from agent_system.multi_turn_rollout import TrajectoryCollector
    from sft_only.webshop.trainer import WebshopSFTOnlyTrainer
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.utils.dataset.rl_dataset import collate_fn

    traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    train_sampler = create_rl_sampler(config.data, train_dataset)

    reward_manager_name = config.reward_model.get("reward_manager", "episode")
    if reward_manager_name == "episode":
        from agent_system.reward_manager import EpisodeRewardManager

        reward_manager_cls = EpisodeRewardManager
    else:
        raise NotImplementedError
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

    return WebshopSFTOnlyTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=None,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        device_name=config.trainer.device,
        traj_collector=traj_collector,
        envs=envs,
        val_envs=val_envs,
    )


if __name__ == "__main__":
    main()
