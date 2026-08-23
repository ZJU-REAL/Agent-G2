from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

from gmsv.alfworld import (
    ExpertTrajectory,
    ExpertTrajectoryStore,
    PrefixPlan,
    _compute_group_thresholds,
    _compute_short_medium_long_thresholds,
    build_step_end_offsets,
    clip_value,
    format_expert_prefix,
    resolve_trial_id_from_gamefile,
)


class CosineHintAlfworldPrefixRuntime:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.cosine_hint_cfg = config.cosine_hint
        self.prefix_cfg = self.cosine_hint_cfg
        self.rng = np.random.default_rng(int(self.cosine_hint_cfg.get("seed", 1)))
        self.store = self._load_expert_store()
        self.loaded_global_step = 0
        self.train_batches_completed = 0
        self._initialized = False

    def _default_expert_json_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "sft_data" / "alfworld_sft_data.json")

    def _expert_json_path(self) -> str:
        configured_path = self.cosine_hint_cfg.get("expert_json_path")
        return str(configured_path) if configured_path else self._default_expert_json_path()

    def _load_expert_store(self) -> ExpertTrajectoryStore:
        json_path = self._expert_json_path()
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        raw_trajectories = payload.get("trajectories", [])
        action_counts = [len(item.get("actions", [])) for item in raw_trajectories]
        difficulty_group_mode = str(self.cosine_hint_cfg.get("difficulty_group_mode", "auto"))
        if difficulty_group_mode == "short_medium_long":
            group_thresholds = _compute_short_medium_long_thresholds(action_counts)
        else:
            num_groups = int(self.cosine_hint_cfg.get("num_difficulty_groups", 5))
            group_thresholds = _compute_group_thresholds(action_counts, num_groups=max(1, num_groups))

        temp_store = ExpertTrajectoryStore({}, group_thresholds)
        store_items: dict[str, ExpertTrajectory] = {}

        for item in raw_trajectories:
            trial_id = str(item["id"])
            raw_actions = item.get("actions", [])
            raw_thinks = item.get("think", [])
            has_thinks = isinstance(raw_thinks, list) and len(raw_thinks) > 0
            actions: list[str] = []
            thinks: list[str] = []
            for action_idx, raw_action in enumerate(raw_actions):
                action = str(raw_action).strip()
                if not action:
                    continue
                actions.append(action)
                if has_thinks:
                    think = str(raw_thinks[action_idx]).strip() if action_idx < len(raw_thinks) else ""
                    thinks.append(think)
            if len(thinks) != len(actions):
                thinks = []

            step_end_offsets, full_prefix_token_ids = build_step_end_offsets(self.tokenizer, actions)
            action_count = len(actions)
            store_items[trial_id] = ExpertTrajectory(
                trial_id=trial_id,
                task_type=str(item.get("task_type", "")),
                actions=actions,
                thinks=thinks,
                action_count=action_count,
                difficulty_group=temp_store.difficulty_group_for_length(action_count),
                step_end_offsets=step_end_offsets,
                full_prefix_token_ids=full_prefix_token_ids,
            )

        return ExpertTrajectoryStore(store_items, group_thresholds)

    def should_apply(self, env_name: str, is_train: bool) -> bool:
        if not bool(self.cosine_hint_cfg.get("enable", False)):
            return False
        if "alfworld" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.cosine_hint_cfg.get("apply_on_train", True))
        return bool(self.cosine_hint_cfg.get("apply_on_validation", False))

    def _current_train_step(self) -> int:
        return int(self.loaded_global_step + self.train_batches_completed)

    def _schedule_total_steps(self) -> int:
        configured_steps = self.cosine_hint_cfg.get("total_training_steps", None)
        if configured_steps is not None:
            return max(int(configured_steps), 1)
        trainer_steps = self.config.trainer.get("total_training_steps", None)
        if trainer_steps is not None:
            return max(int(trainer_steps), 1)
        return max(int(self.config.trainer.get("total_epochs", 300)), 1)

    def _cosine_base_ratio(self, is_train: bool) -> tuple[float, float]:
        if not is_train and self.cosine_hint_cfg.get("validation_ratio", None) is not None:
            ratio = float(self.cosine_hint_cfg.get("validation_ratio"))
            ratio = clip_value(ratio, 0.0, 1.0)
            return 1.0, ratio

        total_steps = self._schedule_total_steps()
        schedule_denominator = max(total_steps - 1, 1)
        progress = min(max(float(self._current_train_step()) / float(schedule_denominator), 0.0), 1.0)
        cosine_value = math.cos(0.5 * math.pi * progress)
        max_ratio = float(self.cosine_hint_cfg.get("max_ratio", 1.0))
        min_ratio = float(self.cosine_hint_cfg.get("min_ratio", 0.0))
        base_ratio = min_ratio + (max_ratio - min_ratio) * cosine_value
        return cosine_value, base_ratio

    def _cosine_ratio(self, is_train: bool) -> tuple[float, float, float]:
        cosine_value, base_ratio = self._cosine_base_ratio(is_train=is_train)
        sigma = float(self.cosine_hint_cfg.get("sigma", 0.0))
        sampled_ratio = float(self.rng.normal(loc=base_ratio, scale=max(sigma, 1e-8))) if sigma > 0.0 else float(base_ratio)
        clipped_ratio = clip_value(sampled_ratio, 0.0, 1.0)
        return clipped_ratio, cosine_value, base_ratio

    def _state_file_path(self, checkpoint_folder: str) -> str:
        return os.path.join(checkpoint_folder, "cosine_hint_runtime_state.json")

    def _resolve_checkpoint_folder(self) -> Optional[str]:
        resume_mode = str(self.config.trainer.resume_mode)
        if resume_mode == "disable":
            return None

        checkpoint_folder = str(self.config.trainer.default_local_dir)
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)

        if resume_mode == "auto":
            return find_latest_ckpt_path(checkpoint_folder)
        if resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
            if not global_step_folder:
                return None
            global_step_folder = str(global_step_folder)
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
            return global_step_folder
        return None

    def _load_persisted_state(self, checkpoint_folder: str) -> bool:
        state_path = self._state_file_path(checkpoint_folder)
        if not os.path.exists(state_path):
            return False

        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.loaded_global_step = int(state.get("loaded_global_step", self.loaded_global_step))
        self.train_batches_completed = int(state.get("train_batches_completed", 0))
        return True

    def ensure_initialized(self) -> None:
        if self._initialized:
            return

        checkpoint_folder = self._resolve_checkpoint_folder()
        if checkpoint_folder and "global_step_" in checkpoint_folder:
            self.loaded_global_step = int(str(checkpoint_folder).split("global_step_")[-1])
        else:
            self.loaded_global_step = 0

        if checkpoint_folder:
            self._load_persisted_state(checkpoint_folder)
        self._initialized = True

    def save_to_checkpoint(self, checkpoint_folder: str) -> None:
        self.ensure_initialized()
        os.makedirs(checkpoint_folder, exist_ok=True)

        state = {
            "loaded_global_step": int(self.loaded_global_step + self.train_batches_completed),
            "train_batches_completed": 0,
        }
        with open(self._state_file_path(checkpoint_folder), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def update_after_train_batch(self, non_tensor_batch: dict[str, Any]) -> dict[str, float]:
        self.ensure_initialized()

        traj_uids = non_tensor_batch.get("traj_uid")
        rollout_ids = traj_uids if traj_uids is not None else non_tensor_batch.get("uid")
        if rollout_ids is None:
            return {}

        episode_rewards = np.asarray(non_tensor_batch.get("episode_rewards", np.zeros(len(rollout_ids), dtype=np.float32)), dtype=np.float32)
        expert_matched = np.asarray(non_tensor_batch.get("cosine_hint_expert_matched", np.ones(len(rollout_ids), dtype=bool)), dtype=bool)
        prefix_steps = np.asarray(non_tensor_batch.get("cosine_hint_prefix_step_count", []), dtype=np.float32)
        hint_ratios = np.asarray(non_tensor_batch.get("cosine_hint_hint_ratio", []), dtype=np.float32)

        grouped_rollout_flags: dict[str, list[float]] = {}
        rollout_expert_matched: dict[str, bool] = {}
        for rollout_id, episode_reward, is_matched in zip(rollout_ids, episode_rewards, expert_matched):
            rollout_key = str(rollout_id)
            grouped_rollout_flags.setdefault(rollout_key, []).append(float(episode_reward > 0.0))
            rollout_expert_matched[rollout_key] = rollout_expert_matched.get(rollout_key, False) or bool(is_matched)

        rollout_accs = []
        matched_rollout_count = 0
        for rollout_id, flags in grouped_rollout_flags.items():
            if not rollout_expert_matched.get(rollout_id, False):
                continue
            matched_rollout_count += 1
            rollout_accs.append(float(max(flags)) if flags else 0.0)

        current_step = self._current_train_step()
        cosine_value, base_ratio = self._cosine_base_ratio(is_train=True)
        batch_accuracy = float(np.mean(np.asarray(rollout_accs, dtype=np.float32))) if rollout_accs else 0.0
        metrics = {
            "cosine_hint/train_step": float(current_step),
            "cosine_hint/total_training_steps": float(self._schedule_total_steps()),
            "cosine_hint/cosine_value": float(cosine_value),
            "cosine_hint/base_ratio": float(base_ratio),
            "cosine_hint/matched_prompt_count": float(matched_rollout_count),
            "cosine_hint/batch_prompt_accuracy": batch_accuracy,
        }
        if prefix_steps.size > 0:
            metrics["cosine_hint/avg_prefix_step_count"] = float(prefix_steps.mean())
            metrics["cosine_hint/max_prefix_step_count"] = float(prefix_steps.max())
        if hint_ratios.size > 0:
            metrics["cosine_hint/avg_hint_ratio"] = float(hint_ratios.mean())

        self.train_batches_completed += 1
        return metrics

    def _empty_plan(self, trial_id: str) -> PrefixPlan:
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

    def _sample_plan(self, trajectory: Optional[ExpertTrajectory], is_train: bool, trial_id: str) -> PrefixPlan:
        if trajectory is None:
            if bool(self.cosine_hint_cfg.get("strict_expert_match", False)):
                raise KeyError(f"No expert trajectory found for ALFWorld trial_id={trial_id}")
            return self._empty_plan(trial_id=trial_id)

        total_steps = trajectory.action_count
        total_tokens = len(trajectory.full_prefix_token_ids)
        group_id = trajectory.difficulty_group
        sampled_ratio, _, base_ratio = self._cosine_ratio(is_train=is_train)

        force_no_prefix = False
        fixed_no_prefix_train = False
        if bool(self.cosine_hint_cfg.get("frozen_no_prefix_baseline", False)):
            force_no_prefix = True
            fixed_no_prefix_train = True
        if is_train and float(self.cosine_hint_cfg.get("train_no_prefix_ratio", 0.0)) > 0.0:
            sampled_no_prefix = bool(self.rng.random() < float(self.cosine_hint_cfg.get("train_no_prefix_ratio", 0.0)))
            force_no_prefix = force_no_prefix or sampled_no_prefix
            fixed_no_prefix_train = fixed_no_prefix_train or sampled_no_prefix

        if force_no_prefix or total_steps <= 0 or total_tokens <= 0:
            return PrefixPlan(
                trial_id=trajectory.trial_id,
                matched=True,
                task_type=trajectory.task_type,
                difficulty_group=group_id,
                mu=base_ratio,
                sigma=float(self.cosine_hint_cfg.get("sigma", 0.0)),
                sampled_ratio=0.0,
                clipped_ratio=0.0,
                prefix_actions=[],
                prefix_thinks=[],
                prefix_step_count=0,
                prefix_token_count=0,
                total_step_count=total_steps,
                total_token_count=total_tokens,
                extended_to_step_end=False,
                fixed_no_prefix_train=fixed_no_prefix_train,
            )

        raw_token_target = int(math.floor(total_tokens * sampled_ratio))
        raw_token_target = max(0, min(raw_token_target, total_tokens))

        if raw_token_target <= 0:
            keep_steps = 0
            extended = False
        elif raw_token_target >= total_tokens:
            keep_steps = total_steps
            extended = False
        else:
            keep_steps = next(
                (idx + 1 for idx, token_end in enumerate(trajectory.step_end_offsets) if token_end >= raw_token_target),
                total_steps,
            )
            extended = trajectory.step_end_offsets[keep_steps - 1] > raw_token_target

        if bool(self.cosine_hint_cfg.get("reserve_model_step", True)) and keep_steps >= total_steps:
            keep_steps = max(total_steps - 1, 0)
            extended = False

        max_budget_steps = max(int(self.config.env.max_steps), 0)
        if bool(self.cosine_hint_cfg.get("reserve_model_step", True)) and max_budget_steps > 0:
            max_budget_steps = max(max_budget_steps - 1, 0)
        if keep_steps > max_budget_steps:
            keep_steps = max_budget_steps
            extended = False

        prefix_actions = trajectory.actions[:keep_steps]
        prefix_thinks = trajectory.thinks[:keep_steps] if trajectory.thinks else []
        prefix_token_count = 0
        if prefix_actions:
            prefix_token_count = len(self.tokenizer.encode(format_expert_prefix(prefix_actions), add_special_tokens=False))

        realized_ratio = float(prefix_token_count / total_tokens) if total_tokens > 0 else 0.0
        return PrefixPlan(
            trial_id=trajectory.trial_id,
            matched=True,
            task_type=trajectory.task_type,
            difficulty_group=group_id,
            mu=base_ratio,
            sigma=float(self.cosine_hint_cfg.get("sigma", 0.0)),
            sampled_ratio=sampled_ratio,
            clipped_ratio=realized_ratio,
            prefix_actions=prefix_actions,
            prefix_thinks=prefix_thinks,
            prefix_step_count=keep_steps,
            prefix_token_count=prefix_token_count,
            total_step_count=total_steps,
            total_token_count=total_tokens,
            extended_to_step_end=extended,
            fixed_no_prefix_train=fixed_no_prefix_train,
        )

    def build_prefix_plans(
        self,
        infos: list[dict[str, Any]],
        is_train: bool,
        group_size: int = 1,
        share_within_group: bool = True,
    ) -> list[PrefixPlan]:
        self.ensure_initialized()
        plans: list[PrefixPlan] = []
        if not share_within_group:
            for info in infos:
                trial_id = resolve_trial_id_from_gamefile(info.get("extra.gamefile"))
                trajectory = self.store.get(trial_id)
                plans.append(self._sample_plan(trajectory=trajectory, is_train=is_train, trial_id=trial_id))
            return plans

        normalized_group_size = max(int(group_size), 1)
        for group_start in range(0, len(infos), normalized_group_size):
            group_infos = infos[group_start : group_start + normalized_group_size]
            trial_id = resolve_trial_id_from_gamefile(group_infos[0].get("extra.gamefile"))
            trajectory = self.store.get(trial_id)
            shared_plan = self._sample_plan(trajectory=trajectory, is_train=is_train, trial_id=trial_id)
            for _ in group_infos:
                plans.append(replace(shared_plan))
        return plans

    def build_batch_metadata(self, plans: list[PrefixPlan]) -> dict[str, np.ndarray]:
        cosine_value, base_ratio = self._cosine_base_ratio(is_train=True)
        return {
            "cosine_hint_trial_id": np.array([plan.trial_id for plan in plans], dtype=object),
            "cosine_hint_expert_matched": np.array([plan.matched for plan in plans], dtype=bool),
            "cosine_hint_task_type": np.array([plan.task_type for plan in plans], dtype=object),
            "cosine_hint_difficulty_group": np.array([plan.difficulty_group for plan in plans], dtype=np.int64),
            "cosine_hint_train_step": np.array([self._current_train_step() for _ in plans], dtype=np.int64),
            "cosine_hint_total_training_steps": np.array([self._schedule_total_steps() for _ in plans], dtype=np.int64),
            "cosine_hint_cosine_value": np.array([cosine_value for _ in plans], dtype=np.float32),
            "cosine_hint_base_ratio": np.array([base_ratio for _ in plans], dtype=np.float32),
            "cosine_hint_requested_ratio": np.array([plan.sampled_ratio for plan in plans], dtype=np.float32),
            "cosine_hint_hint_ratio": np.array([plan.clipped_ratio for plan in plans], dtype=np.float32),
            "cosine_hint_prefix_step_count": np.array([plan.prefix_step_count for plan in plans], dtype=np.int64),
            "cosine_hint_prefix_token_count": np.array([plan.prefix_token_count for plan in plans], dtype=np.int64),
            "cosine_hint_total_step_count": np.array([plan.total_step_count for plan in plans], dtype=np.int64),
            "cosine_hint_total_token_count": np.array([plan.total_token_count for plan in plans], dtype=np.int64),
            "cosine_hint_extended_to_step_end": np.array([plan.extended_to_step_end for plan in plans], dtype=bool),
            "cosine_hint_fixed_no_prefix_train": np.array([plan.fixed_no_prefix_train for plan in plans], dtype=bool),
        }


def build_cosine_hint_runtime(tokenizer, config) -> Optional[CosineHintAlfworldPrefixRuntime]:
    if not bool(config.get("cosine_hint", {}).get("enable", False)):
        return None
    if "alfworld" not in str(config.env.env_name).lower():
        return None
    return CosineHintAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
