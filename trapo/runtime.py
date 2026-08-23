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
from gmsv.webshop import (
    WebshopExpertTrajectoryStore,
    _compute_webshop_three_group_thresholds,
    item_to_target_key,
    load_webshop_trajectories,
    normalize_target_key,
    target_key_to_string,
)


@dataclass
class TRAPOPrefixPlan:
    trial_id: str
    gamefile: str
    matched: bool
    task_type: str
    difficulty_group: int
    micro_group_index: int
    prompt_index: int
    triggered: bool
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


class TRAPOAlfworldPrefixRuntime:
    """ALFWorld expert-prefix runtime for TRAPO micro-group sampling.

    The runtime follows the existing prefix replay contract used by GMSV, but
    prefix lengths are supplied by the TRAPO micro-group controller instead of
    sampled from a Gaussian model.
    """

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.trapo_cfg = config.get("trapo", {})
        self.prefix_cfg = self.trapo_cfg
        self.store = self._load_expert_store()
        self.micro_group_index = 0
        self.active_group_size = 1
        self.ratio_by_group_index: dict[int, float] = {}
        self.triggered_by_group_index: dict[int, bool] = {}
        self.gamefile_by_group_index: dict[int, str] = {}
        self.last_plans: list[TRAPOPrefixPlan] = []
        self.last_metadata: dict[str, Any] = {}

    def _default_expert_json_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "sft_data" / "alfworld_sft_data.json")

    def _expert_json_path(self) -> str:
        configured_path = self.trapo_cfg.get("expert_json_path")
        if configured_path:
            return str(configured_path)
        gmsv_path = self.config.get("gmsv", {}).get("expert_json_path")
        return str(gmsv_path) if gmsv_path else self._default_expert_json_path()

    def _load_expert_store(self) -> ExpertTrajectoryStore:
        with open(self._expert_json_path(), "r", encoding="utf-8") as f:
            payload = json.load(f)

        raw_trajectories = payload.get("trajectories", [])
        action_counts = [len(item.get("actions", [])) for item in raw_trajectories]
        difficulty_group_mode = str(self.trapo_cfg.get("difficulty_group_mode", "auto"))
        if difficulty_group_mode == "short_medium_long":
            group_thresholds = _compute_short_medium_long_thresholds(action_counts)
        else:
            num_groups = int(self.trapo_cfg.get("num_difficulty_groups", 5))
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
        if not bool(self.trapo_cfg.get("enable", False)):
            return False
        if "alfworld" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.trapo_cfg.get("apply_on_train", True))
        return bool(self.trapo_cfg.get("apply_on_validation", False))

    def configure_micro_group(
        self,
        *,
        micro_group_index: int,
        active_group_size: int,
        ratio_by_group_index: dict[int, float],
        triggered_by_group_index: Optional[dict[int, bool]] = None,
        gamefile_by_group_index: Optional[dict[int, str]] = None,
    ) -> None:
        self.micro_group_index = int(micro_group_index)
        self.active_group_size = max(int(active_group_size), 1)
        self.ratio_by_group_index = {int(key): clip_value(float(value), 0.0, 1.0) for key, value in ratio_by_group_index.items()}
        self.triggered_by_group_index = {int(key): bool(value) for key, value in (triggered_by_group_index or {}).items()}
        self.gamefile_by_group_index = {int(key): str(value) for key, value in (gamefile_by_group_index or {}).items()}

    def _empty_plan(
        self,
        *,
        trial_id: str,
        gamefile: str,
        prompt_index: int,
        requested_ratio: float,
        triggered: bool,
        matched: bool = False,
    ) -> TRAPOPrefixPlan:
        return TRAPOPrefixPlan(
            trial_id=trial_id,
            gamefile=gamefile,
            matched=matched,
            task_type="",
            difficulty_group=1,
            micro_group_index=self.micro_group_index,
            prompt_index=prompt_index,
            triggered=triggered,
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

    def _max_prefix_steps(self, total_steps: int) -> int:
        allow_full_prefix = bool(self.trapo_cfg.get("allow_full_prefix", False))
        reserve_model_step = bool(self.trapo_cfg.get("reserve_model_step", True))
        if allow_full_prefix and not reserve_model_step:
            return max(min(total_steps, int(self.config.env.max_steps)), 0)
        return max(min(total_steps - 1, int(self.config.env.max_steps) - 1), 0)

    def _build_plan(
        self,
        *,
        trajectory: ExpertTrajectory | None,
        trial_id: str,
        gamefile: str,
        prompt_index: int,
    ) -> TRAPOPrefixPlan:
        requested_ratio = clip_value(float(self.ratio_by_group_index.get(prompt_index, 0.0)), 0.0, 1.0)
        triggered = bool(self.triggered_by_group_index.get(prompt_index, requested_ratio > 0.0))
        if trajectory is None:
            if bool(self.trapo_cfg.get("strict_expert_match", False)):
                raise KeyError(f"No expert trajectory found for ALFWorld trial_id={trial_id}")
            return self._empty_plan(
                trial_id=trial_id,
                gamefile=gamefile,
                prompt_index=prompt_index,
                requested_ratio=requested_ratio,
                triggered=triggered,
                matched=False,
            )

        total_steps = trajectory.action_count
        total_tokens = len(trajectory.full_prefix_token_ids)
        if requested_ratio <= 0.0 or total_steps <= 0 or total_tokens <= 0:
            plan = self._empty_plan(
                trial_id=trajectory.trial_id,
                gamefile=gamefile,
                prompt_index=prompt_index,
                requested_ratio=requested_ratio,
                triggered=triggered,
                matched=True,
            )
            plan.task_type = trajectory.task_type
            plan.difficulty_group = trajectory.difficulty_group
            plan.total_step_count = total_steps
            plan.total_token_count = total_tokens
            return plan

        raw_token_target = int(math.floor(total_tokens * requested_ratio))
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

        keep_steps = min(keep_steps, self._max_prefix_steps(total_steps))
        prefix_actions = trajectory.actions[:keep_steps]
        prefix_thinks = trajectory.thinks[:keep_steps] if trajectory.thinks else []
        prefix_token_count = 0
        if prefix_actions:
            prefix_token_count = len(self.tokenizer.encode(format_expert_prefix(prefix_actions), add_special_tokens=False))

        actual_ratio = clip_value(float(prefix_token_count / total_tokens), 0.0, 1.0) if total_tokens > 0 else 0.0
        return TRAPOPrefixPlan(
            trial_id=trajectory.trial_id,
            gamefile=gamefile,
            matched=True,
            task_type=trajectory.task_type,
            difficulty_group=trajectory.difficulty_group,
            micro_group_index=self.micro_group_index,
            prompt_index=prompt_index,
            triggered=triggered,
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
    ) -> list[TRAPOPrefixPlan]:
        plans: list[TRAPOPrefixPlan] = []
        normalized_group_size = self.active_group_size
        for env_idx, info in enumerate(infos):
            prompt_index = env_idx // normalized_group_size
            source_info = infos[prompt_index * normalized_group_size] if share_within_group else info
            gamefile = self.gamefile_by_group_index.get(prompt_index) or str(source_info.get("extra.gamefile") or "")
            trial_id = resolve_trial_id_from_gamefile(gamefile)
            trajectory = self.store.get(trial_id)
            plans.append(
                self._build_plan(
                    trajectory=trajectory,
                    trial_id=trial_id,
                    gamefile=gamefile,
                    prompt_index=prompt_index,
                )
            )
        self.last_plans = plans
        return plans

    def build_batch_metadata(self, plans: list[TRAPOPrefixPlan]) -> dict[str, np.ndarray]:
        full_prefix = np.array(
            [plan.matched and plan.total_step_count > 0 and plan.prefix_step_count >= plan.total_step_count for plan in plans],
            dtype=bool,
        )
        return {
            "trapo_trial_id": np.array([plan.trial_id for plan in plans], dtype=object),
            "trapo_gamefile": np.array([plan.gamefile for plan in plans], dtype=object),
            "trapo_expert_matched": np.array([plan.matched for plan in plans], dtype=bool),
            "trapo_task_type": np.array([plan.task_type for plan in plans], dtype=object),
            "trapo_difficulty_group": np.array([plan.difficulty_group for plan in plans], dtype=np.int64),
            "trapo_micro_group_index": np.array([plan.micro_group_index for plan in plans], dtype=np.int64),
            "trapo_prompt_index": np.array([plan.prompt_index for plan in plans], dtype=np.int64),
            "trapo_triggered": np.array([plan.triggered for plan in plans], dtype=bool),
            "trapo_requested_ratio": np.array([plan.requested_ratio for plan in plans], dtype=np.float32),
            "trapo_hint_ratio": np.array([plan.clipped_ratio for plan in plans], dtype=np.float32),
            "trapo_prefix_step_count": np.array([plan.prefix_step_count for plan in plans], dtype=np.int64),
            "trapo_prefix_token_count": np.array([plan.prefix_token_count for plan in plans], dtype=np.int64),
            "trapo_total_step_count": np.array([plan.total_step_count for plan in plans], dtype=np.int64),
            "trapo_total_token_count": np.array([plan.total_token_count for plan in plans], dtype=np.int64),
            "trapo_full_prefix": full_prefix,
            "trapo_extended_to_step_end": np.array([plan.extended_to_step_end for plan in plans], dtype=bool),
            "gmsv_full_prefix": full_prefix,
        }

    def update_after_train_batch(self, non_tensor_batch: dict[str, Any]) -> dict[str, float]:
        micro_groups = np.asarray(non_tensor_batch.get("trapo_micro_group_index", []), dtype=np.int64)
        if micro_groups.size == 0:
            return {}

        metrics: dict[str, float] = {}
        requested = np.asarray(non_tensor_batch.get("trapo_requested_ratio", []), dtype=np.float32)
        used = np.asarray(non_tensor_batch.get("trapo_hint_ratio", []), dtype=np.float32)
        triggered = np.asarray(non_tensor_batch.get("trapo_triggered", []), dtype=bool)
        matched = np.asarray(non_tensor_batch.get("trapo_expert_matched", []), dtype=bool)

        if requested.size:
            metrics["trapo/train/requested_ratio_mean"] = float(requested.mean())
        if used.size:
            metrics["trapo/train/hint_ratio_mean"] = float(used.mean())
        if triggered.size:
            metrics["trapo/train/triggered_ratio"] = float(triggered.mean())
        if matched.size:
            metrics["trapo/train/expert_matched_ratio"] = float(matched.mean())

        for micro_group in sorted(set(int(value) for value in micro_groups.tolist())):
            mask = micro_groups == micro_group
            if requested.size:
                metrics[f"trapo/train/micro_group_{micro_group}/requested_ratio_mean"] = float(requested[mask].mean())
            if triggered.size:
                metrics[f"trapo/train/micro_group_{micro_group}/triggered_ratio"] = float(triggered[mask].mean())
            rewards = np.asarray(non_tensor_batch.get("episode_rewards", []), dtype=np.float32)
            if rewards.size == micro_groups.size:
                metrics[f"trapo/train/micro_group_{micro_group}/success_rate"] = float(np.mean(rewards[mask] > 0.0))
        return metrics

    def save_to_checkpoint(self, checkpoint_folder: str) -> None:
        os.makedirs(checkpoint_folder, exist_ok=True)
        state_path = os.path.join(checkpoint_folder, "trapo_runtime_state.json")
        payload = {"last_metadata": self.last_metadata}
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


class TRAPOWebshopPrefixRuntime(TRAPOAlfworldPrefixRuntime):
    def _default_expert_json_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "sft_data" / "webshop_sft_data.json")

    def _load_expert_store(self) -> WebshopExpertTrajectoryStore:
        json_path = self._expert_json_path()
        raw_trajectories = load_webshop_trajectories(json_path)
        action_counts = [len(item.get("actions", [])) for item in raw_trajectories]
        difficulty_group_mode = str(self.trapo_cfg.get("difficulty_group_mode", "webshop_3"))
        if difficulty_group_mode in {"webshop_3", "webshop_three"}:
            group_thresholds = _compute_webshop_three_group_thresholds(action_counts)
        elif difficulty_group_mode == "short_medium_long":
            group_thresholds = _compute_short_medium_long_thresholds(action_counts)
        else:
            num_groups = int(self.trapo_cfg.get("num_difficulty_groups", 3))
            group_thresholds = _compute_group_thresholds(action_counts, num_groups=max(1, num_groups))

        temp_store = WebshopExpertTrajectoryStore({}, group_thresholds)
        store_items = {}
        skipped = 0
        for item in raw_trajectories:
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

            target_key = item_to_target_key(item)
            if target_key is None:
                skipped += 1
                continue
            if target_key in store_items:
                continue

            step_end_offsets, full_prefix_token_ids = build_step_end_offsets(self.tokenizer, actions)
            action_count = len(actions)
            trial_id = str(item.get("id") or target_key_to_string(target_key))
            store_items[target_key] = ExpertTrajectory(
                trial_id=trial_id,
                task_type=str(item.get("task_type", "webshop")),
                actions=actions,
                thinks=thinks,
                action_count=action_count,
                difficulty_group=temp_store.difficulty_group_for_length(action_count),
                step_end_offsets=step_end_offsets,
                full_prefix_token_ids=full_prefix_token_ids,
            )

        if skipped:
            print(f"[TRAPO WebShop] skipped {skipped} expert trajectories without target key")
        print(f"[TRAPO WebShop] loaded {len(store_items)} unique expert target keys from {json_path}")
        return WebshopExpertTrajectoryStore(store_items, group_thresholds)

    def should_apply(self, env_name: str, is_train: bool) -> bool:
        if not bool(self.trapo_cfg.get("enable", False)):
            return False
        if "webshop" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.trapo_cfg.get("apply_on_train", True))
        return bool(self.trapo_cfg.get("apply_on_validation", False))

    def _build_plan(
        self,
        *,
        trajectory: ExpertTrajectory | None,
        trial_id: str,
        gamefile: str,
        prompt_index: int,
    ) -> TRAPOPrefixPlan:
        if trajectory is None and bool(self.trapo_cfg.get("strict_expert_match", False)):
            raise KeyError(f"No expert trajectory found for WebShop target_key={trial_id}")
        return super()._build_plan(
            trajectory=trajectory,
            trial_id=trial_id,
            gamefile=gamefile,
            prompt_index=prompt_index,
        )

    def build_prefix_plans(
        self,
        infos: list[dict[str, Any]],
        is_train: bool,
        group_size: int = 1,
        share_within_group: bool = True,
    ) -> list[TRAPOPrefixPlan]:
        plans: list[TRAPOPrefixPlan] = []
        normalized_group_size = self.active_group_size
        for env_idx, info in enumerate(infos):
            prompt_index = env_idx // normalized_group_size
            source_info = infos[prompt_index * normalized_group_size] if share_within_group else info
            configured_key = self.gamefile_by_group_index.get(prompt_index)
            target_key = normalize_target_key(configured_key) or normalize_target_key(source_info.get("webshop_target_key"))
            target_key_text = target_key_to_string(target_key)
            trajectory = self.store.get(target_key)
            trial_id = trajectory.trial_id if trajectory is not None else target_key_text
            plans.append(
                self._build_plan(
                    trajectory=trajectory,
                    trial_id=trial_id,
                    gamefile=target_key_text,
                    prompt_index=prompt_index,
                )
            )
        self.last_plans = plans
        return plans
