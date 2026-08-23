from __future__ import annotations

import uuid
import json
import os
from pprint import pprint
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from gmsv.alfworld import PrefixPlan, resolve_trial_id_from_gamefile
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, reduce_metrics


class SFTOnlyTrainer(RayPPOTrainer):
    """Train only on expert replay prefix-SFT pairs, without model rollout or RL loss."""

    def _validate_config(self):
        super()._validate_config()
        if int(self.config.env.rollout.n) != 1:
            raise ValueError("SFT-only training requires env.rollout.n=1 so each problem uses one expert trajectory.")
        if not bool(self.config.gmsv.get("enable", False)):
            raise ValueError("SFT-only training currently requires gmsv.enable=true.")
        if not bool(self.config.gmsv.prefix_sft.get("enable", False)):
            raise ValueError("SFT-only training requires gmsv.prefix_sft.enable=true.")
        if bool(self.config.actor_rollout_ref.actor.get("use_kl_loss", False)):
            print("Warning: actor KL loss is enabled, but SFT-only batches mask policy tokens so KL contributes zero.")

    def _full_expert_plan(self, info: Dict[str, Any]) -> PrefixPlan:
        prefix_runtime = getattr(self.traj_collector, "gmsv_runtime", None)
        if prefix_runtime is None:
            raise RuntimeError("SFT-only trainer requires a GMSV runtime.")
        prefix_runtime.ensure_initialized()

        trial_id = resolve_trial_id_from_gamefile(info.get("extra.gamefile"))
        trajectory = prefix_runtime.store.get(trial_id)
        if trajectory is None:
            if bool(prefix_runtime.gmsv_cfg.get("strict_expert_match", False)):
                raise KeyError(f"No expert trajectory found for trial_id={trial_id}")
            return PrefixPlan(
                trial_id=trial_id,
                matched=False,
                task_type="",
                difficulty_group=1,
                mu=0.0,
                sigma=0.0,
                sampled_ratio=0.0,
                clipped_ratio=0.0,
                prefix_actions=[],
                prefix_thinks=[],
                prefix_step_count=0,
                prefix_token_count=0,
                total_step_count=0,
                total_token_count=0,
                extended_to_step_end=False,
                fixed_no_prefix_train=False,
            )

        keep_steps = min(int(trajectory.action_count), max(int(self.config.env.max_steps), 0))
        prefix_actions = trajectory.actions[:keep_steps]
        prefix_thinks = trajectory.thinks[:keep_steps] if trajectory.thinks else []
        return PrefixPlan(
            trial_id=trajectory.trial_id,
            matched=True,
            task_type=trajectory.task_type,
            difficulty_group=trajectory.difficulty_group,
            mu=1.0,
            sigma=0.0,
            sampled_ratio=1.0,
            clipped_ratio=1.0,
            prefix_actions=prefix_actions,
            prefix_thinks=prefix_thinks,
            prefix_step_count=keep_steps,
            prefix_token_count=len(trajectory.full_prefix_token_ids),
            total_step_count=trajectory.action_count,
            total_token_count=len(trajectory.full_prefix_token_ids),
            extended_to_step_end=False,
            fixed_no_prefix_train=False,
        )

    def _build_full_expert_plans(self, infos: list[dict[str, Any]]) -> list[PrefixPlan]:
        return [self._full_expert_plan(info) for info in infos]

    def _success_from_prefix_infos(self, batch_size: int) -> dict[str, np.ndarray]:
        terminal_infos = getattr(self.envs, "prefix_terminal_infos", None) or []
        success_values = []
        for idx in range(batch_size):
            info = terminal_infos[idx] if idx < len(terminal_infos) else None
            won = bool(info.get("won", False)) if isinstance(info, dict) else False
            success_values.append(float(won))
        return {"success_rate": np.asarray(success_values, dtype=np.float32)}

    def _prepare_gen_batch(self, batch: DataProto) -> DataProto:
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "env_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("env_kwargs")

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": False,
            "temperature": float(self.config.actor_rollout_ref.rollout.temperature),
        }
        return gen_batch

    def _collect_expert_sft_batch(
        self,
        gen_batch: DataProto,
        global_step: int,
    ) -> tuple[Optional[DataProto], Optional[dict[str, Any]], dict[str, float]]:
        env_kwargs = gen_batch.non_tensor_batch.pop("env_kwargs", None)
        obs, infos = self.envs.reset(kwargs=env_kwargs)
        batch_size = len(obs["text"]) if obs.get("text") is not None else len(obs["image"])
        if len(gen_batch.batch) != batch_size:
            raise AssertionError(f"gen_batch size {len(gen_batch.batch)} does not match env batch size {batch_size}")

        should_dump = self.traj_collector._should_dump_rollout_detail(
            dump_dir=self.config.trainer.get("train_rollout_detail_data_dir", None),
            global_step=global_step,
        )
        prefix_plans = self._build_full_expert_plans(infos)
        _, prefix_rewards, prefix_lengths, _ = self.envs.replay_prefix_plans(
            prefix_plans,
            collect_sft=True,
            collect_detail=should_dump,
        )

        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        prefix_sft_records = getattr(self.envs, "prefix_sft_records", None)
        sft_batch = self.traj_collector._build_prefix_sft_batch(
            gen_batch=gen_batch,
            prefix_sft_records=prefix_sft_records,
            traj_uid=traj_uid,
        )

        rollout_payload = None
        if should_dump:
            prefix_detail_records = getattr(self.envs, "prefix_detail_records", None)
            rollout_payload = self.traj_collector._build_rollout_details_payload(
                total_batch_list=[[] for _ in range(batch_size)],
                episode_rewards=prefix_rewards,
                episode_lengths=prefix_lengths,
                success=self._success_from_prefix_infos(batch_size),
                traj_uid=traj_uid,
                tool_callings=np.zeros(batch_size, dtype=np.float32),
                global_step=global_step,
                is_train=True,
                prefix_detail_records=prefix_detail_records,
            )
            dump_dir = self.config.trainer.get("train_rollout_detail_data_dir", None)
            self.traj_collector._dump_rollout_details_payload(
                payload=rollout_payload,
                dump_dir=dump_dir,
                global_step=global_step,
            )
            self._dump_sft_pairs_outputs(
                rollout_payload=rollout_payload,
                dump_dir=dump_dir,
                global_step=global_step,
            )

        metrics = {
            "sft_only/expert_matched": float(np.mean([float(plan.matched) for plan in prefix_plans])) if prefix_plans else 0.0,
            "sft_only/prefix_steps": float(np.sum(prefix_lengths)),
            "sft_only/episode_reward_mean": float(np.mean(prefix_rewards)) if len(prefix_rewards) else 0.0,
        }
        return sft_batch, rollout_payload, metrics

    def _prepare_sft_only_actor_batch(self, sft_batch: DataProto, metrics: dict) -> DataProto:
        if "prefix_sft_padding_mask" not in sft_batch.batch.keys():
            sft_batch.batch["prefix_sft_padding_mask"] = torch.zeros(len(sft_batch), dtype=torch.bool)
        if self.config.actor_rollout_ref.actor.get("use_kl_loss", False) and "ref_log_prob" not in sft_batch.batch.keys():
            sft_batch.batch["ref_log_prob"] = torch.zeros_like(sft_batch.batch["responses"], dtype=torch.float32)

        pad_size = (-len(sft_batch)) % self.actor_rollout_wg.world_size
        if pad_size:
            pad_batch = sft_batch[:1].repeat(pad_size)
            response_length = pad_batch.batch["responses"].shape[-1]
            pad_batch.batch["advantages"] = torch.zeros_like(pad_batch.batch["advantages"])
            pad_batch.batch["old_log_probs"] = torch.zeros_like(pad_batch.batch["old_log_probs"])
            pad_batch.batch["prefix_sft_mask"] = torch.zeros((pad_size, response_length), dtype=torch.bool)
            pad_batch.batch["prefix_sft_step_weight"] = torch.zeros(pad_size, dtype=torch.float32)
            pad_batch.batch["prefix_sft_padding_mask"] = torch.ones(pad_size, dtype=torch.bool)
            if "ref_log_prob" in pad_batch.batch.keys():
                pad_batch.batch["ref_log_prob"] = torch.zeros_like(pad_batch.batch["ref_log_prob"])
            sft_batch = DataProto.concat([sft_batch, pad_batch])

        prefix_sft_cfg = self.config.gmsv.get("prefix_sft", {})
        sft_batch.meta_info = dict(sft_batch.meta_info)
        sft_batch.meta_info["temperature"] = float(self.config.actor_rollout_ref.rollout.temperature)
        sft_batch.meta_info["global_token_num"] = torch.sum(sft_batch.batch["attention_mask"], dim=-1).tolist()
        sft_batch.meta_info["prefix_sft_enabled"] = True
        sft_batch.meta_info["prefix_sft_loss_coef"] = float(prefix_sft_cfg.get("loss_coef", 1.0))
        sft_batch.meta_info["prefix_sft_loss_type"] = str(prefix_sft_cfg.get("loss_type", "ce"))
        sft_batch.meta_info["prefix_sft_clip_low"] = float(prefix_sft_cfg.get("clip_low", 0.1))
        sft_batch.meta_info["multi_turn"] = False

        metrics["prefix_sft/num_samples"] = float((~sft_batch.batch["prefix_sft_padding_mask"].bool()).sum().item())
        metrics["prefix_sft/num_tokens"] = float(sft_batch.batch["prefix_sft_mask"].sum().item())
        return sft_batch

    def _dump_sft_pairs_outputs(
        self,
        rollout_payload: Optional[dict[str, Any]],
        dump_dir: str | None,
        global_step: int,
    ) -> None:
        if not self.traj_collector._should_dump_rollout_detail(dump_dir=dump_dir, global_step=global_step):
            return
        payload = self.traj_collector._build_sft_pairs_payload(rollout_payload)
        if payload is None or payload["num_pairs"] <= 0:
            return

        sft_pairs_dir = os.path.join(dump_dir, "sft_pairs")
        os.makedirs(sft_pairs_dir, exist_ok=True)

        step_output_path = os.path.join(sft_pairs_dir, f"global_step_{global_step}.json")
        with open(step_output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        all_pairs_path = os.path.join(sft_pairs_dir, "all_sft_pairs.jsonl")
        with open(all_pairs_path, "a", encoding="utf-8") as f:
            for pair in payload["pairs"]:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        print(f"Dumped prefix SFT training pairs to {step_output_path}")
        print(f"Appended prefix SFT training pairs to {all_pairs_path}")

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="SFT-only Training Progress")
        self.global_steps += 1

        last_val_metrics = None
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                batch = DataProto.from_single_dict(batch_dict)
                gen_batch = self._prepare_gen_batch(batch)
                is_last_step = self.global_steps >= self.total_training_steps

                sft_batch, _, collect_metrics = self._collect_expert_sft_batch(
                    gen_batch=gen_batch,
                    global_step=self.global_steps,
                )
                metrics.update(collect_metrics)

                if sft_batch is None or len(sft_batch) == 0:
                    metrics["sft_only/skipped_no_sft_pairs"] = 1.0
                else:
                    actor_batch = self._prepare_sft_only_actor_batch(sft_batch, metrics)
                    actor_output = self.actor_rollout_wg.update_actor(actor_batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })

                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    val_metrics = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                logger.log(data=metrics, step=self.global_steps)

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    self._save_checkpoint()

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    pprint("SFT-only training finished")
                    progress_bar.close()
                    return
