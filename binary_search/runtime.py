from __future__ import annotations

import json
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
class BinarySearchPrefixPlan:
    trial_id: str
    gamefile: str
    matched: bool
    task_type: str
    difficulty_group: int
    requested_step_count: int
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


class BinarySearchAlfworldPrefixRuntime:
    """Fixed-step ALFWorld prefix runtime used by online binary search.

    The class intentionally mirrors GMSV's expert-prefix formatting and replay
    contract, but chooses prefix length from an explicit per-gamefile/per-trial
    step map instead of sampling mu/sigma.
    """

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.binary_search_cfg = config.get("binary_search", {})
        self.prefix_cfg = self.binary_search_cfg
        self.default_step_count = 0
        self.default_step_strategy: str | None = None
        self.step_by_group_index: dict[int, int] = {}
        self.step_by_gamefile: dict[str, int] = {}
        self.step_by_trial_id: dict[str, int] = {}
        self.store = self._load_expert_store()
        self.last_metadata: dict[str, Any] = {}
        self.last_plans: list[BinarySearchPrefixPlan] = []

    def _default_expert_json_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "sft_data" / "alfworld_sft_data.json")

    def _expert_json_path(self) -> str:
        configured_path = self.binary_search_cfg.get("expert_json_path")
        if configured_path:
            return str(configured_path)
        gmsv_path = self.config.get("gmsv", {}).get("expert_json_path")
        return str(gmsv_path) if gmsv_path else self._default_expert_json_path()

    def _load_expert_store(self) -> ExpertTrajectoryStore:
        with open(self._expert_json_path(), "r", encoding="utf-8") as f:
            payload = json.load(f)

        raw_trajectories = payload.get("trajectories", [])
        action_counts = [len(item.get("actions", [])) for item in raw_trajectories]
        difficulty_group_mode = str(self.binary_search_cfg.get("difficulty_group_mode", self.config.get("gmsv", {}).get("difficulty_group_mode", "auto")))
        if difficulty_group_mode == "short_medium_long":
            group_thresholds = _compute_short_medium_long_thresholds(action_counts)
        else:
            num_groups = int(self.binary_search_cfg.get("num_difficulty_groups", self.config.get("gmsv", {}).get("num_difficulty_groups", 5)))
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

    def configure_steps(
        self,
        *,
        default_step_count: int = 0,
        step_by_group_index: Optional[dict[int, int]] = None,
        step_by_gamefile: Optional[dict[str, int]] = None,
        step_by_trial_id: Optional[dict[str, int]] = None,
    ) -> None:
        self.default_step_count = max(int(default_step_count), 0)
        self.default_step_strategy = None
        self.step_by_group_index = {int(key): max(int(value), 0) for key, value in (step_by_group_index or {}).items()}
        self.step_by_gamefile = {str(key): max(int(value), 0) for key, value in (step_by_gamefile or {}).items()}
        self.step_by_trial_id = {str(key): max(int(value), 0) for key, value in (step_by_trial_id or {}).items()}

    def configure_midpoint_steps(self) -> None:
        self.default_step_count = 0
        self.default_step_strategy = "midpoint"
        self.step_by_group_index = {}
        self.step_by_gamefile = {}
        self.step_by_trial_id = {}

    def should_apply(self, env_name: str, is_train: bool) -> bool:
        if not bool(self.binary_search_cfg.get("enable", False)):
            return False
        if "alfworld" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.binary_search_cfg.get("apply_on_train", True))
        return bool(self.binary_search_cfg.get("apply_on_validation", False))

    def _empty_plan(self, trial_id: str, gamefile: str, matched: bool = False) -> BinarySearchPrefixPlan:
        return BinarySearchPrefixPlan(
            trial_id=trial_id,
            gamefile=gamefile,
            matched=matched,
            task_type="",
            difficulty_group=1,
            requested_step_count=0,
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

    def _explicit_step_count(self, trial_id: str, gamefile: str, group_index: int | None = None) -> int | None:
        if group_index is not None and group_index in self.step_by_group_index:
            return self.step_by_group_index[group_index]
        if gamefile and gamefile in self.step_by_gamefile:
            return self.step_by_gamefile[gamefile]
        if trial_id and trial_id in self.step_by_trial_id:
            return self.step_by_trial_id[trial_id]
        return None

    def _requested_step_count(
        self,
        trajectory: ExpertTrajectory,
        trial_id: str,
        gamefile: str,
        group_index: int | None = None,
    ) -> int:
        explicit_step_count = self._explicit_step_count(trial_id, gamefile, group_index=group_index)
        if explicit_step_count is not None:
            return explicit_step_count
        if self.default_step_strategy == "midpoint":
            upper = min(max(int(trajectory.action_count) - 1, 0), max(int(self.config.env.max_steps) - 1, 0))
            return int(upper // 2)
        return self.default_step_count

    def _build_plan(self, trajectory: ExpertTrajectory | None, trial_id: str, gamefile: str, group_index: int | None = None) -> BinarySearchPrefixPlan:
        if trajectory is None:
            return self._empty_plan(trial_id=trial_id, gamefile=gamefile, matched=False)

        total_steps = trajectory.action_count
        total_tokens = len(trajectory.full_prefix_token_ids)
        requested_step_count = self._requested_step_count(trajectory, trajectory.trial_id, gamefile, group_index=group_index)

        if total_steps <= 0 or total_tokens <= 0:
            plan = self._empty_plan(trial_id=trajectory.trial_id, gamefile=gamefile, matched=True)
            plan.task_type = trajectory.task_type
            plan.difficulty_group = trajectory.difficulty_group
            plan.requested_step_count = requested_step_count
            plan.total_step_count = total_steps
            plan.total_token_count = total_tokens
            return plan

        keep_steps = min(requested_step_count, max(total_steps - 1, 0))
        max_budget_steps = max(int(self.config.env.max_steps) - 1, 0)
        keep_steps = min(keep_steps, max_budget_steps)

        prefix_actions = trajectory.actions[:keep_steps]
        prefix_thinks = trajectory.thinks[:keep_steps] if trajectory.thinks else []
        prefix_token_count = 0
        if prefix_actions:
            prefix_token_count = len(self.tokenizer.encode(format_expert_prefix(prefix_actions), add_special_tokens=False))
        step_ratio = clip_value(float(keep_steps / total_steps), 0.0, 1.0) if total_steps > 0 else 0.0

        return BinarySearchPrefixPlan(
            trial_id=trajectory.trial_id,
            gamefile=gamefile,
            matched=True,
            task_type=trajectory.task_type,
            difficulty_group=trajectory.difficulty_group,
            requested_step_count=requested_step_count,
            sampled_ratio=step_ratio,
            clipped_ratio=step_ratio,
            prefix_actions=prefix_actions,
            prefix_thinks=prefix_thinks,
            prefix_step_count=keep_steps,
            prefix_token_count=prefix_token_count,
            total_step_count=total_steps,
            total_token_count=total_tokens,
            extended_to_step_end=False,
            fixed_no_prefix_train=False,
        )

    def build_prefix_plans(
        self,
        infos: list[dict[str, Any]],
        is_train: bool,
        group_size: int = 1,
        share_within_group: bool = True,
    ) -> list[BinarySearchPrefixPlan]:
        plans: list[BinarySearchPrefixPlan] = []
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
            plans.append(self._build_plan(trajectory=trajectory, trial_id=trial_id, gamefile=gamefile, group_index=env_idx // normalized_group_size))
        self.last_plans = plans
        return plans

    def build_batch_metadata(self, plans: list[BinarySearchPrefixPlan]) -> dict[str, np.ndarray]:
        group_size = max(int(self.config.env.rollout.n), 1)
        return {
            "binary_search_mode": np.array(["step" for _ in plans], dtype=object),
            "binary_search_trial_id": np.array([plan.trial_id for plan in plans], dtype=object),
            "binary_search_gamefile": np.array([plan.gamefile for plan in plans], dtype=object),
            "binary_search_requested_step_count": np.array([plan.requested_step_count for plan in plans], dtype=np.int64),
            "binary_search_prefix_step_count": np.array([plan.prefix_step_count for plan in plans], dtype=np.int64),
            "binary_search_total_step_count": np.array([plan.total_step_count for plan in plans], dtype=np.int64),
            "binary_search_prefix_token_count": np.array([plan.prefix_token_count for plan in plans], dtype=np.int64),
            "binary_search_total_token_count": np.array([plan.total_token_count for plan in plans], dtype=np.int64),
            "binary_search_ratio": np.array([plan.clipped_ratio for plan in plans], dtype=np.float32),
            "binary_search_matched": np.array([plan.matched for plan in plans], dtype=bool),
            "binary_search_group_index": np.array([idx // group_size for idx in range(len(plans))], dtype=np.int64),
        }

    def update_after_train_batch(self, non_tensor_batch: dict[str, Any]) -> dict[str, float]:
        group_indices = np.asarray(non_tensor_batch.get("binary_search_group_index", []))
        if group_indices.size == 0:
            return {}

        metrics: dict[str, float] = {}
        for key in (
            "binary_search_selected_step_count",
            "binary_search_selected_success_count",
            "binary_search_eval_count",
            "binary_search_selected_gap",
        ):
            values = np.asarray(non_tensor_batch.get(key, []), dtype=np.float32)
            if values.size > 0:
                metrics[f"binary_search/train/{key.replace('binary_search_', '')}_mean"] = float(values.mean())

        statuses = np.asarray(non_tensor_batch.get("binary_search_status", []), dtype=object)
        if statuses.size > 0:
            unique_statuses = sorted({str(status) for status in statuses})
            denom = max(float(statuses.size), 1.0)
            for status in unique_statuses:
                metrics[f"binary_search/train/status_{status}_ratio"] = float(np.sum(statuses == status) / denom)
        return metrics

    def save_to_checkpoint(self, checkpoint_folder: str) -> None:
        os.makedirs(checkpoint_folder, exist_ok=True)
        state_path = os.path.join(checkpoint_folder, "binary_search_runtime_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"last_metadata": self.last_metadata}, f, ensure_ascii=False, indent=2)


def build_binary_search_runtime(tokenizer, config) -> Optional[BinarySearchAlfworldPrefixRuntime]:
    if not bool(config.get("binary_search", {}).get("enable", False)):
        return None
    if "alfworld" not in str(config.env.env_name).lower():
        return None
    return BinarySearchAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
