from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gmsv.alfworld import ExpertTrajectory
from gmsv.webshop import (
    WebshopExpertTrajectoryStore,
    _compute_webshop_three_group_thresholds,
    item_to_target_key,
    load_webshop_trajectories,
    normalize_target_key,
    target_key_to_string,
)
from liner_hint.alfworld import (
    LinerHintAlfworldPrefixRuntime,
    PrefixPlan,
    build_step_end_offsets,
)


class LinerHintWebshopPrefixRuntime(LinerHintAlfworldPrefixRuntime):
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
            print(f"[LinerHint WebShop] skipped {skipped} expert trajectories without target key")
        print(f"[LinerHint WebShop] loaded {len(store_items)} unique expert target keys from {json_path}")
        return WebshopExpertTrajectoryStore(store_items, group_thresholds)

    def should_apply(self, env_name: str, is_train: bool) -> bool:
        if not bool(self.liner_hint_cfg.get("enable", False)):
            return False
        if "webshop" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.liner_hint_cfg.get("apply_on_train", True))
        return bool(self.liner_hint_cfg.get("apply_on_validation", False))

    def _sample_plan(self, trajectory: Optional[ExpertTrajectory], is_train: bool, trial_id: str) -> PrefixPlan:
        if trajectory is not None:
            return super()._sample_plan(trajectory=trajectory, is_train=is_train, trial_id=trial_id)
        if bool(self.liner_hint_cfg.get("strict_expert_match", False)):
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
                        "liner_hint.share_prefix_within_group=true; got "
                        f"{target_key_to_string(target_key)} and {target_key_to_string(group_target_key)}"
                    )
            trajectory = self.store.get(target_key)
            trial_id = trajectory.trial_id if trajectory is not None else target_key_to_string(target_key)
            shared_plan = self._sample_plan(trajectory=trajectory, is_train=is_train, trial_id=trial_id)
            for _ in group_infos:
                plans.append(replace(shared_plan))
        return plans
