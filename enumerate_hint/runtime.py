from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gmsv.alfworld import (
    ExpertTrajectory,
    ExpertTrajectoryStore,
    _compute_group_thresholds,
    _compute_short_medium_long_thresholds,
    build_step_end_offsets,
    clip_value,
    format_expert_prefix,
    resolve_trial_id_from_gamefile,
)


@dataclass
class EnumerateHintPrefixPlan:
    trial_id: str
    gamefile: str
    matched: bool
    task_type: str
    difficulty_group: int
    requested_ratio: float
    sampled_ratio: float
    clipped_ratio: float
    prefix_actions: list[str]
    prefix_thinks: list[str]
    prefix_step_count: int
    prefix_token_count: int
    total_step_count: int
    total_token_count: int
    extended_to_step_end: bool
    fixed_no_prefix_train: bool


class EnumerateHintAlfworldPrefixRuntime:
    """Fixed-ratio ALFWorld prefix runtime for online enumerate-hint search."""

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.enumerate_hint_cfg = config.get("enumerate_hint", {})
        self.prefix_cfg = self.enumerate_hint_cfg
        self.default_ratio = 0.0
        self.ratio_by_group_index: dict[int, float] = {}
        self.ratio_by_gamefile: dict[str, float] = {}
        self.ratio_by_trial_id: dict[str, float] = {}
        self.default_step_count: int | None = None
        self.step_count_by_group_index: dict[int, int] = {}
        self.step_count_by_gamefile: dict[str, int] = {}
        self.step_count_by_trial_id: dict[str, int] = {}
        self.store = self._load_expert_store()
        self.last_metadata: dict[str, Any] = {}
        self.last_plans: list[EnumerateHintPrefixPlan] = []

    def _default_expert_json_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "sft_data" / "alfworld_sft_data.json")

    def _expert_json_path(self) -> str:
        configured_path = self.enumerate_hint_cfg.get("expert_json_path")
        if configured_path:
            return str(configured_path)
        gmsv_path = self.config.get("gmsv", {}).get("expert_json_path")
        return str(gmsv_path) if gmsv_path else self._default_expert_json_path()

    def _load_expert_store(self) -> ExpertTrajectoryStore:
        with open(self._expert_json_path(), "r", encoding="utf-8") as f:
            payload = json.load(f)

        raw_trajectories = payload.get("trajectories", [])
        action_counts = [len(item.get("actions", [])) for item in raw_trajectories]
        difficulty_group_mode = str(
            self.enumerate_hint_cfg.get(
                "difficulty_group_mode",
                self.config.get("gmsv", {}).get("difficulty_group_mode", "auto"),
            )
        )
        if difficulty_group_mode == "short_medium_long":
            group_thresholds = _compute_short_medium_long_thresholds(action_counts)
        else:
            num_groups = int(
                self.enumerate_hint_cfg.get(
                    "num_difficulty_groups",
                    self.config.get("gmsv", {}).get("num_difficulty_groups", 5),
                )
            )
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

    def configure_ratios(
        self,
        *,
        default_ratio: float = 0.0,
        ratio_by_group_index: Optional[dict[int, float]] = None,
        ratio_by_gamefile: Optional[dict[str, float]] = None,
        ratio_by_trial_id: Optional[dict[str, float]] = None,
    ) -> None:
        self.default_ratio = clip_value(float(default_ratio), 0.0, 1.0)
        self.ratio_by_group_index = {
            int(key): clip_value(float(value), 0.0, 1.0)
            for key, value in (ratio_by_group_index or {}).items()
        }
        self.ratio_by_gamefile = {
            str(key): clip_value(float(value), 0.0, 1.0)
            for key, value in (ratio_by_gamefile or {}).items()
        }
        self.ratio_by_trial_id = {
            str(key): clip_value(float(value), 0.0, 1.0)
            for key, value in (ratio_by_trial_id or {}).items()
        }
        self.default_step_count = None
        self.step_count_by_group_index = {}
        self.step_count_by_gamefile = {}
        self.step_count_by_trial_id = {}

    def configure_step_counts(
        self,
        *,
        default_step_count: int | None = None,
        step_count_by_group_index: Optional[dict[int, int]] = None,
        step_count_by_gamefile: Optional[dict[str, int]] = None,
        step_count_by_trial_id: Optional[dict[str, int]] = None,
    ) -> None:
        self.default_step_count = int(default_step_count) if default_step_count is not None else None
        self.step_count_by_group_index = {
            int(key): max(int(value), 0)
            for key, value in (step_count_by_group_index or {}).items()
        }
        self.step_count_by_gamefile = {
            str(key): max(int(value), 0)
            for key, value in (step_count_by_gamefile or {}).items()
        }
        self.step_count_by_trial_id = {
            str(key): max(int(value), 0)
            for key, value in (step_count_by_trial_id or {}).items()
        }
        self.default_ratio = 0.0
        self.ratio_by_group_index = {}
        self.ratio_by_gamefile = {}
        self.ratio_by_trial_id = {}

    def should_apply(self, env_name: str, is_train: bool) -> bool:
        if not bool(self.enumerate_hint_cfg.get("enable", False)):
            return False
        if "alfworld" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.enumerate_hint_cfg.get("apply_on_train", True))
        return bool(self.enumerate_hint_cfg.get("apply_on_validation", False))

    def _empty_plan(self, trial_id: str, gamefile: str, requested_ratio: float = 0.0, matched: bool = False) -> EnumerateHintPrefixPlan:
        return EnumerateHintPrefixPlan(
            trial_id=trial_id,
            gamefile=gamefile,
            matched=matched,
            task_type="",
            difficulty_group=1,
            requested_ratio=requested_ratio,
            sampled_ratio=requested_ratio,
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

    def _requested_ratio(self, trial_id: str, gamefile: str, group_index: int | None = None) -> float:
        if group_index is not None and group_index in self.ratio_by_group_index:
            return self.ratio_by_group_index[group_index]
        if gamefile and gamefile in self.ratio_by_gamefile:
            return self.ratio_by_gamefile[gamefile]
        if trial_id and trial_id in self.ratio_by_trial_id:
            return self.ratio_by_trial_id[trial_id]
        return self.default_ratio

    def _requested_step_count(self, trial_id: str, gamefile: str, group_index: int | None = None) -> int | None:
        if group_index is not None and group_index in self.step_count_by_group_index:
            return self.step_count_by_group_index[group_index]
        if gamefile and gamefile in self.step_count_by_gamefile:
            return self.step_count_by_gamefile[gamefile]
        if trial_id and trial_id in self.step_count_by_trial_id:
            return self.step_count_by_trial_id[trial_id]
        return self.default_step_count

    def _keep_steps_for_ratio(self, trajectory: ExpertTrajectory, ratio: float) -> tuple[int, bool]:
        total_steps = trajectory.action_count
        total_tokens = len(trajectory.full_prefix_token_ids)
        clipped_ratio = clip_value(float(ratio), 0.0, float(self.enumerate_hint_cfg.get("max_length", 1.0)))
        clipped_ratio = clip_value(clipped_ratio, 0.0, 1.0)
        if total_steps <= 0 or total_tokens <= 0 or clipped_ratio <= 0.0:
            return 0, False

        raw_token_target = int(math.floor(total_tokens * clipped_ratio))
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

        keep_steps = min(keep_steps, max(total_steps - 1, 0))
        keep_steps = min(keep_steps, max(int(self.config.env.max_steps) - 1, 0))
        if keep_steps <= 0:
            return 0, False
        return keep_steps, extended

    def _keep_steps_exact(self, trajectory: ExpertTrajectory, step_count: int) -> int:
        max_steps = max(trajectory.action_count - 1, 0)
        max_steps = min(max_steps, max(int(self.config.env.max_steps) - 1, 0))
        return max(0, min(int(step_count), max_steps))

    def step_count_for_ratio(self, trial_id: str, gamefile: str, ratio: float) -> int:
        trajectory = self.store.get(trial_id)
        if trajectory is None:
            resolved_trial_id = resolve_trial_id_from_gamefile(gamefile)
            trajectory = self.store.get(resolved_trial_id)
        if trajectory is None:
            return 0
        keep_steps, _ = self._keep_steps_for_ratio(trajectory, ratio)
        return int(keep_steps)

    def _build_plan(
        self,
        trajectory: ExpertTrajectory | None,
        trial_id: str,
        gamefile: str,
        group_index: int | None = None,
    ) -> EnumerateHintPrefixPlan:
        requested_step_count = self._requested_step_count(trial_id, gamefile, group_index=group_index)
        requested_ratio = self._requested_ratio(trial_id, gamefile, group_index=group_index)
        if trajectory is None:
            if bool(self.enumerate_hint_cfg.get("strict_expert_match", False)):
                raise KeyError(f"No expert trajectory found for ALFWorld trial_id={trial_id}")
            return self._empty_plan(trial_id=trial_id, gamefile=gamefile, requested_ratio=requested_ratio, matched=False)

        total_steps = trajectory.action_count
        total_tokens = len(trajectory.full_prefix_token_ids)
        if total_steps <= 0 or total_tokens <= 0:
            plan = self._empty_plan(
                trial_id=trajectory.trial_id,
                gamefile=gamefile,
                requested_ratio=requested_ratio,
                matched=True,
            )
            plan.task_type = trajectory.task_type
            plan.difficulty_group = trajectory.difficulty_group
            plan.total_step_count = total_steps
            plan.total_token_count = total_tokens
            return plan

        if requested_step_count is not None:
            keep_steps = self._keep_steps_exact(trajectory, requested_step_count)
            extended = False
            requested_ratio = clip_value(float(requested_step_count / total_steps), 0.0, 1.0)
        else:
            keep_steps, extended = self._keep_steps_for_ratio(trajectory, requested_ratio)
        prefix_actions = trajectory.actions[:keep_steps]
        prefix_thinks = trajectory.thinks[:keep_steps] if trajectory.thinks else []
        prefix_token_count = 0
        if prefix_actions:
            prefix_token_count = len(self.tokenizer.encode(format_expert_prefix(prefix_actions), add_special_tokens=False))
        actual_ratio = clip_value(float(keep_steps / total_steps), 0.0, 1.0) if total_steps > 0 else 0.0

        return EnumerateHintPrefixPlan(
            trial_id=trajectory.trial_id,
            gamefile=gamefile,
            matched=True,
            task_type=trajectory.task_type,
            difficulty_group=trajectory.difficulty_group,
            requested_ratio=requested_ratio,
            sampled_ratio=requested_ratio,
            clipped_ratio=actual_ratio,
            prefix_actions=prefix_actions,
            prefix_thinks=prefix_thinks,
            prefix_step_count=keep_steps,
            prefix_token_count=prefix_token_count,
            total_step_count=total_steps,
            total_token_count=total_tokens,
            extended_to_step_end=extended,
            fixed_no_prefix_train=False,
        )

    def build_prefix_plans(
        self,
        infos: list[dict[str, Any]],
        is_train: bool,
        group_size: int = 1,
        share_within_group: bool = True,
    ) -> list[EnumerateHintPrefixPlan]:
        plans: list[EnumerateHintPrefixPlan] = []
        normalized_group_size = max(int(group_size), 1)
        for env_idx, info in enumerate(infos):
            if share_within_group:
                group_start = (env_idx // normalized_group_size) * normalized_group_size
                source_info = infos[group_start]
            else:
                source_info = info
            gamefile = str(source_info.get("extra.gamefile") or "")
            trial_id = resolve_trial_id_from_gamefile(gamefile)
            trajectory = self.store.get(trial_id)
            plans.append(
                self._build_plan(
                    trajectory=trajectory,
                    trial_id=trial_id,
                    gamefile=gamefile,
                    group_index=env_idx // normalized_group_size,
                )
            )
        self.last_plans = plans
        return plans

    def build_batch_metadata(self, plans: list[EnumerateHintPrefixPlan]) -> dict[str, np.ndarray]:
        group_size = max(int(self.config.env.rollout.n), 1)
        return {
            "enumerate_hint_trial_id": np.array([plan.trial_id for plan in plans], dtype=object),
            "enumerate_hint_gamefile": np.array([plan.gamefile for plan in plans], dtype=object),
            "enumerate_hint_requested_ratio": np.array([plan.requested_ratio for plan in plans], dtype=np.float32),
            "enumerate_hint_ratio": np.array([plan.clipped_ratio for plan in plans], dtype=np.float32),
            "enumerate_hint_prefix_step_count": np.array([plan.prefix_step_count for plan in plans], dtype=np.int64),
            "enumerate_hint_total_step_count": np.array([plan.total_step_count for plan in plans], dtype=np.int64),
            "enumerate_hint_prefix_token_count": np.array([plan.prefix_token_count for plan in plans], dtype=np.int64),
            "enumerate_hint_total_token_count": np.array([plan.total_token_count for plan in plans], dtype=np.int64),
            "enumerate_hint_matched": np.array([plan.matched for plan in plans], dtype=bool),
            "enumerate_hint_group_index": np.array([idx // group_size for idx in range(len(plans))], dtype=np.int64),
        }

    def update_after_train_batch(self, non_tensor_batch: dict[str, Any]) -> dict[str, float]:
        group_indices = np.asarray(non_tensor_batch.get("enumerate_hint_group_index", []))
        if group_indices.size == 0:
            return {}

        metrics: dict[str, float] = {}
        for key in (
            "enumerate_hint_selected_ratio",
            "enumerate_hint_selected_step_count",
            "enumerate_hint_selected_success_count",
            "enumerate_hint_eval_count",
            "enumerate_hint_duplicate_skip_count",
        ):
            values = np.asarray(non_tensor_batch.get(key, []), dtype=np.float32)
            if values.size > 0:
                metrics[f"enumerate_hint/train/{key.replace('enumerate_hint_', '')}_mean"] = float(values.mean())

        statuses = np.asarray(non_tensor_batch.get("enumerate_hint_status", []), dtype=object)
        if statuses.size > 0:
            unique_statuses = sorted({str(status) for status in statuses})
            denom = max(float(statuses.size), 1.0)
            for status in unique_statuses:
                metrics[f"enumerate_hint/train/status_{status}_ratio"] = float(np.sum(statuses == status) / denom)
        return metrics

    def save_to_checkpoint(self, checkpoint_folder: str) -> None:
        os.makedirs(checkpoint_folder, exist_ok=True)
        state_path = os.path.join(checkpoint_folder, "enumerate_hint_runtime_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"last_metadata": self.last_metadata}, f, ensure_ascii=False, indent=2)


def build_enumerate_hint_runtime(tokenizer, config) -> Optional[EnumerateHintAlfworldPrefixRuntime]:
    if not bool(config.get("enumerate_hint", {}).get("enable", False)):
        return None
    if "alfworld" not in str(config.env.env_name).lower():
        return None
    return EnumerateHintAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
