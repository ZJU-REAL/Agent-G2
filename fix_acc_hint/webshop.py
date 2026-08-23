from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from fix_acc_hint.alfworld import FixAccHintAlfworldPrefixRuntime
from gmsv.alfworld import (
    ExpertTrajectory,
    PrefixPlan,
    build_step_end_offsets,
    format_expert_prefix,
)
from gmsv.webshop import (
    WebshopExpertTrajectoryStore,
    _compute_webshop_three_group_thresholds,
    item_to_target_key,
    load_webshop_trajectories,
    normalize_target_key,
    target_key_to_string,
)


class FixAccHintWebshopPrefixRuntime(FixAccHintAlfworldPrefixRuntime):
    def _default_expert_json_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "sft_data" / "webshop_sft_data.json")

    def _load_expert_store(self) -> WebshopExpertTrajectoryStore:
        json_path = self._expert_json_path()
        raw_trajectories = load_webshop_trajectories(json_path)
        action_counts = [len(item.get("actions", [])) for item in raw_trajectories]
        group_thresholds = _compute_webshop_three_group_thresholds(action_counts)
        temp_store = WebshopExpertTrajectoryStore({}, group_thresholds)
        store_items: dict[Any, ExpertTrajectory] = {}
        skipped = 0

        for item in raw_trajectories:
            raw_actions = item.get("actions", [])
            raw_thinks = item.get("think", [])
            has_thinks = isinstance(raw_thinks, list) and len(raw_thinks) > 0
            actions = []
            thinks = []
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

            target_key = item_to_target_key(item)
            if target_key is None:
                skipped += 1
                continue
            if target_key in store_items:
                continue

            step_end_offsets, full_prefix_token_ids = build_step_end_offsets(self.tokenizer, actions)
            action_count = len(actions)
            difficulty_group = temp_store.difficulty_group_for_length(action_count)
            trial_id = str(item.get("id") or target_key_to_string(target_key))
            store_items[target_key] = ExpertTrajectory(
                trial_id=trial_id,
                task_type=str(item.get("task_type", "webshop")),
                actions=actions,
                thinks=thinks,
                action_count=action_count,
                difficulty_group=difficulty_group,
                step_end_offsets=step_end_offsets,
                full_prefix_token_ids=full_prefix_token_ids,
            )

        if skipped:
            print(f"[FixAccHint WebShop] skipped {skipped} expert trajectories without target key")
        print(f"[FixAccHint WebShop] loaded {len(store_items)} unique expert target keys from {json_path}")
        return WebshopExpertTrajectoryStore(store_items, group_thresholds)

    def should_apply(self, env_name: str, is_train: bool) -> bool:
        if not bool(self.fix_acc_hint_cfg.get("enable", False)):
            return False
        if "webshop" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.fix_acc_hint_cfg.get("apply_on_train", True))
        return bool(self.fix_acc_hint_cfg.get("apply_on_validation", False))

    def _update_drop_count(self, batch_accuracy: float, matched_rollout_count: int) -> bool:
        if matched_rollout_count <= 0:
            return False
        if batch_accuracy <= float(self.fix_acc_hint_cfg.get("acc_threshold", 0.5)):
            return False

        step_delta = max(int(self.fix_acc_hint_cfg.get("step_delta", 1)), 1)
        self.drop_count += step_delta
        max_drop_count = self.fix_acc_hint_cfg.get("max_drop_count", None)
        if max_drop_count is not None:
            self.drop_count = min(self.drop_count, max(int(max_drop_count), 0))
        return True

    def _sample_plan(self, trajectory: Optional[ExpertTrajectory], is_train: bool, trial_id: str) -> PrefixPlan:
        if trajectory is None:
            if bool(self.fix_acc_hint_cfg.get("strict_expert_match", False)):
                raise KeyError(f"No expert trajectory found for WebShop target_key={trial_id}")
            return PrefixPlan(
                trial_id=trial_id,
                matched=False,
                task_type="webshop",
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

        total_steps = trajectory.action_count
        total_tokens = len(trajectory.full_prefix_token_ids)
        reserve_model_step = bool(self.fix_acc_hint_cfg.get("reserve_model_step", True))
        max_keep_steps = max(total_steps - 1, 0) if reserve_model_step else total_steps

        min_prefix_steps = max(int(self.fix_acc_hint_cfg.get("min_prefix_steps", 0)), 0)
        drop_count = max(int(self.drop_count), 0) if is_train else max(int(self.fix_acc_hint_cfg.get("validation_drop_count", 0)), 0)
        keep_steps = max(max_keep_steps - drop_count, min_prefix_steps)
        keep_steps = min(keep_steps, max_keep_steps)

        max_budget_steps = max(int(self.config.env.max_steps), 0)
        if reserve_model_step and max_budget_steps > 0:
            max_budget_steps = max(max_budget_steps - 1, 0)
        if keep_steps > max_budget_steps:
            keep_steps = max_budget_steps

        force_no_prefix = False
        fixed_no_prefix_train = False
        if bool(self.fix_acc_hint_cfg.get("frozen_no_prefix_baseline", False)):
            force_no_prefix = True
            fixed_no_prefix_train = True
        if is_train and float(self.fix_acc_hint_cfg.get("train_no_prefix_ratio", 0.0)) > 0.0:
            sampled_no_prefix = bool(self.rng.random() < float(self.fix_acc_hint_cfg.get("train_no_prefix_ratio", 0.0)))
            force_no_prefix = force_no_prefix or sampled_no_prefix
            fixed_no_prefix_train = fixed_no_prefix_train or sampled_no_prefix

        if force_no_prefix or total_steps <= 0 or total_tokens <= 0:
            keep_steps = 0

        prefix_actions = trajectory.actions[:keep_steps]
        prefix_thinks = trajectory.thinks[:keep_steps] if trajectory.thinks else []
        prefix_token_count = 0
        if prefix_actions:
            prefix_token_count = len(self.tokenizer.encode(format_expert_prefix(prefix_actions), add_special_tokens=False))

        hint_ratio = float(keep_steps) / float(total_steps) if total_steps > 0 else 0.0
        return PrefixPlan(
            trial_id=trajectory.trial_id,
            matched=True,
            task_type=trajectory.task_type,
            difficulty_group=trajectory.difficulty_group,
            mu=float(max_keep_steps),
            sigma=0.0,
            sampled_ratio=hint_ratio,
            clipped_ratio=hint_ratio,
            prefix_actions=prefix_actions,
            prefix_thinks=prefix_thinks,
            prefix_step_count=keep_steps,
            prefix_token_count=prefix_token_count,
            total_step_count=total_steps,
            total_token_count=total_tokens,
            extended_to_step_end=False,
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
                target_key = normalize_target_key(info.get("webshop_target_key"))
                trajectory = self.store.get(target_key)
                trial_id = trajectory.trial_id if trajectory is not None else target_key_to_string(target_key)
                plans.append(self._sample_plan(trajectory=trajectory, is_train=is_train, trial_id=trial_id))
            return plans

        normalized_group_size = max(int(group_size), 1)
        for group_start in range(0, len(infos), normalized_group_size):
            group_infos = infos[group_start : group_start + normalized_group_size]
            target_key = normalize_target_key(group_infos[0].get("webshop_target_key"))
            for info in group_infos[1:]:
                group_target_key = normalize_target_key(info.get("webshop_target_key"))
                if group_target_key != target_key:
                    raise ValueError(
                        "WebShop grouped rollouts must share one target key when "
                        "fix_acc_hint.share_prefix_within_group=true; got "
                        f"{target_key_to_string(target_key)} and {target_key_to_string(group_target_key)}"
                    )
            trajectory = self.store.get(target_key)
            trial_id = trajectory.trial_id if trajectory is not None else target_key_to_string(target_key)
            shared_plan = self._sample_plan(trajectory=trajectory, is_train=is_train, trial_id=trial_id)
            for _ in group_infos:
                plans.append(replace(shared_plan))
        return plans
