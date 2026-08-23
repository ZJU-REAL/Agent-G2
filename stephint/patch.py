from __future__ import annotations

import torch

from .rollout import stephint_multi_turn_loop


_PATCHED = False


def _patch_trajectory_collector() -> None:
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    if getattr(TrajectoryCollector, "_stephint_patched", False):
        return

    original_init = TrajectoryCollector.__init__
    original_multi_turn_loop = TrajectoryCollector.multi_turn_loop

    def patched_init(self, config, tokenizer, processor=None):
        if bool(config.get("step_hint", {}).get("enable", False)):
            enabled_others = [
                name
                for name in (
                    "gmsv",
                    "binary_search",
                    "enumerate_hint",
                    "cosine_hint",
                    "liner_hint",
                    "fix_step_hint",
                    "fix_acc_hint",
                )
                if bool(config.get(name, {}).get("enable", False))
            ]
            if enabled_others:
                raise ValueError(f"step_hint must run alone; also enabled: {enabled_others}")

        original_init(self, config=config, tokenizer=tokenizer, processor=processor)
        self.stephint_runtime = None
        if bool(config.get("step_hint", {}).get("enable", False)):
            from stephint import build_stephint_runtime

            self.stephint_runtime = build_stephint_runtime(tokenizer=tokenizer, config=config)
            self.prefix_runtime = self.stephint_runtime
            self.prefix_runtime_name = "step_hint"

    def patched_multi_turn_loop(
        self,
        gen_batch,
        actor_rollout_wg,
        envs,
        is_train=True,
        prefix_runtime_override=None,
        global_step=None,
        rollout_detail_dump_dir=None,
        collect_rollout_details=False,
    ):
        runtime = getattr(self, "stephint_runtime", None)
        if (
            runtime is not None
            and prefix_runtime_override is None
            and runtime.should_apply(env_name=self.config.env.env_name, is_train=is_train)
        ):
            return stephint_multi_turn_loop(
                collector=self,
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
                prefix_runtime_override=prefix_runtime_override,
                global_step=global_step,
                rollout_detail_dump_dir=rollout_detail_dump_dir,
                collect_rollout_details=collect_rollout_details,
            )
        return original_multi_turn_loop(
            self,
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            is_train=is_train,
            prefix_runtime_override=prefix_runtime_override,
            global_step=global_step,
            rollout_detail_dump_dir=rollout_detail_dump_dir,
            collect_rollout_details=collect_rollout_details,
        )

    TrajectoryCollector.__init__ = patched_init
    TrajectoryCollector._stephint_original_multi_turn_loop = original_multi_turn_loop
    TrajectoryCollector.multi_turn_loop = patched_multi_turn_loop
    TrajectoryCollector._stephint_patched = True


def _patch_grpo_advantage() -> None:
    import verl.trainer.ppo.ray_trainer as ray_trainer
    from verl.trainer.ppo import core_algos
    from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_response_mask

    if getattr(ray_trainer, "_stephint_advantage_patched", False):
        return

    original_compute_advantage = ray_trainer.compute_advantage

    def patched_compute_advantage(
        data,
        adv_estimator,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        multi_turn=False,
        norm_adv_by_std_in_grpo=True,
        step_advantage_w=1.0,
        gigpo_mode="mean_std_norm",
        gigpo_enable_similarity=False,
        gigpo_similarity_thresh=0.95,
        **kwargs,
    ):
        has_stephint = "step_hint_protected_mask" in data.batch
        if has_stephint and adv_estimator == AdvantageEstimator.GRPO:
            if "response_mask" not in data.batch:
                data.batch["response_mask"] = compute_response_mask(data)
            grpo_calculation_mask = data.batch["response_mask"]
            if multi_turn:
                response_length = grpo_calculation_mask.size(1)
                grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]

            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                traj_index=data.non_tensor_batch["traj_uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                compute_mean_std_cross_steps=False,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
        else:
            data = original_compute_advantage(
                data,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                multi_turn=multi_turn,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                step_advantage_w=step_advantage_w,
                gigpo_mode=gigpo_mode,
                gigpo_enable_similarity=gigpo_enable_similarity,
                gigpo_similarity_thresh=gigpo_similarity_thresh,
                **kwargs,
            )

        if has_stephint and "advantages" in data.batch:
            mask = data.batch["step_hint_protected_mask"].bool()
            advantages = data.batch["advantages"]
            if mask.size(-1) != advantages.size(-1):
                mask = mask[:, -advantages.size(-1):]
            data.batch["advantages"] = torch.where(
                mask & (advantages < 0),
                torch.zeros_like(advantages),
                advantages,
            )
        return data

    ray_trainer.compute_advantage = patched_compute_advantage
    ray_trainer._stephint_advantage_patched = True


def apply_stephint_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _patch_trajectory_collector()
    _patch_grpo_advantage()
    _PATCHED = True
