from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gmsv.alfworld import (
    ExpertTrajectory,
    ExpertTrajectoryStore,
    _compute_group_thresholds,
    _compute_short_medium_long_thresholds,
    build_step_end_offsets,
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
class StepHintPlan:
    trial_id: str
    gamefile: str
    matched: bool
    task_type: str
    difficulty_group: int
    level: int
    slot_type: str
    prefix_actions: list[str]
    prefix_thinks: list[str]
    prefix_step_count: int
    total_step_count: int
    prefix_token_count: int
    total_token_count: int
    hint_boundaries: list[int]
    fixed_no_prefix_train: bool = False

    @property
    def is_reference(self) -> bool:
        return self.slot_type == "reference"


class StepHintAlfworldRuntime:
    """Strict StepHint plan builder for ALFWorld.

    A plan does not mean "prompt prefix"; it describes teacher-forced expert
    environment steps that should be inserted into the RL trajectory.
    """

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.cfg = config.get("step_hint", {})
        self.store = self._load_expert_store()
        self.last_plans: list[StepHintPlan] = []
        self.last_metadata: dict[str, Any] = {}

    def _default_expert_json_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "sft_data" / "alfworld_sft_data.json")

    def _expert_json_path(self) -> str:
        configured_path = self.cfg.get("expert_json_path")
        if configured_path:
            return str(configured_path)
        return self._default_expert_json_path()

    def _load_expert_store(self) -> ExpertTrajectoryStore:
        with open(self._expert_json_path(), "r", encoding="utf-8") as f:
            payload = json.load(f)

        raw_trajectories = payload.get("trajectories", [])
        action_counts = [len(item.get("actions", [])) for item in raw_trajectories]
        difficulty_group_mode = str(self.cfg.get("difficulty_group_mode", "short_medium_long"))
        if difficulty_group_mode == "short_medium_long":
            group_thresholds = _compute_short_medium_long_thresholds(action_counts)
        else:
            num_groups = int(self.cfg.get("num_difficulty_groups", 5))
            group_thresholds = _compute_group_thresholds(action_counts, num_groups=max(1, num_groups))

        temp_store = ExpertTrajectoryStore({}, group_thresholds)
        trajectories: dict[str, ExpertTrajectory] = {}
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
            trajectories[trial_id] = ExpertTrajectory(
                trial_id=trial_id,
                task_type=str(item.get("task_type", "")),
                actions=actions,
                thinks=thinks,
                action_count=action_count,
                difficulty_group=temp_store.difficulty_group_for_length(action_count),
                step_end_offsets=step_end_offsets,
                full_prefix_token_ids=full_prefix_token_ids,
            )

        return ExpertTrajectoryStore(trajectories, group_thresholds)

    def should_apply(self, env_name: str, is_train: bool) -> bool:
        if not bool(self.cfg.get("enable", False)):
            return False
        if "alfworld" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.cfg.get("apply_on_train", True))
        return bool(self.cfg.get("apply_on_validation", False))

    def expected_group_size(self) -> int:
        partition_count = max(int(self.cfg.get("partition_count", 4)), 2)
        khint = max(int(self.cfg.get("khint", 2)), 0)
        kunhint = max(int(self.cfg.get("kunhint", 5)), 0)
        include_reference = bool(self.cfg.get("include_reference", True))
        return int(kunhint + khint * (partition_count - 1) + (1 if include_reference else 0))

    def _hint_boundaries(self, total_steps: int) -> list[int]:
        partition_count = max(int(self.cfg.get("partition_count", 4)), 2)
        effective_steps = min(int(total_steps), max(int(self.config.env.max_steps), 0))
        if effective_steps <= 1:
            return []
        boundaries = [
            int(math.ceil(effective_steps * level / partition_count))
            for level in range(1, partition_count)
        ]
        deduped = sorted({boundary for boundary in boundaries if 0 < boundary < effective_steps})
        max_prefix_steps = max(effective_steps - 1, 0)
        capped = {min(boundary, max_prefix_steps) for boundary in deduped if min(boundary, max_prefix_steps) > 0}
        return sorted(capped)

    def _slot_specs(self, trajectory: ExpertTrajectory | None) -> list[tuple[str, int, int]]:
        kunhint = max(int(self.cfg.get("kunhint", 5)), 0)
        khint = max(int(self.cfg.get("khint", 2)), 0)
        partition_count = max(int(self.cfg.get("partition_count", 4)), 2)
        include_reference = bool(self.cfg.get("include_reference", True))

        total_steps = trajectory.action_count if trajectory is not None else 0
        boundaries = self._hint_boundaries(total_steps)
        specs: list[tuple[str, int, int]] = [("unhinted", 0, 0) for _ in range(kunhint)]

        for level in range(1, partition_count):
            keep_steps = boundaries[level - 1] if level - 1 < len(boundaries) else 0
            slot_type = "hint" if keep_steps > 0 else "unhinted"
            specs.extend((slot_type, level if keep_steps > 0 else 0, keep_steps) for _ in range(khint))

        if include_reference:
            reference_steps = min(total_steps, max(int(self.config.env.max_steps), 0))
            specs.append(("reference", partition_count, reference_steps))
        return specs

    def _empty_plan(self, trial_id: str, gamefile: str, slot_type: str, level: int) -> StepHintPlan:
        return StepHintPlan(
            trial_id=trial_id,
            gamefile=gamefile,
            matched=False,
            task_type="",
            difficulty_group=1,
            level=level,
            slot_type=slot_type,
            prefix_actions=[],
            prefix_thinks=[],
            prefix_step_count=0,
            total_step_count=0,
            prefix_token_count=0,
            total_token_count=0,
            hint_boundaries=[],
        )

    def _build_plan(
        self,
        trajectory: ExpertTrajectory | None,
        trial_id: str,
        gamefile: str,
        slot_type: str,
        level: int,
        keep_steps: int,
    ) -> StepHintPlan:
        if trajectory is None:
            if bool(self.cfg.get("strict_expert_match", False)):
                raise KeyError(f"No StepHint expert trajectory found for ALFWorld trial_id={trial_id}")
            return self._empty_plan(trial_id=trial_id, gamefile=gamefile, slot_type="unhinted", level=0)

        total_steps = trajectory.action_count
        keep_steps = max(0, min(int(keep_steps), total_steps, int(self.config.env.max_steps)))
        if slot_type != "reference":
            keep_steps = min(keep_steps, max(total_steps - 1, 0), max(int(self.config.env.max_steps) - 1, 0))

        prefix_actions = trajectory.actions[:keep_steps]
        prefix_thinks = trajectory.thinks[:keep_steps] if trajectory.thinks else []
        prefix_token_count = 0
        if prefix_actions:
            prefix_token_count = len(self.tokenizer.encode(format_expert_prefix(prefix_actions), add_special_tokens=False))

        return StepHintPlan(
            trial_id=trajectory.trial_id,
            gamefile=gamefile,
            matched=True,
            task_type=trajectory.task_type,
            difficulty_group=trajectory.difficulty_group,
            level=level,
            slot_type=slot_type,
            prefix_actions=prefix_actions,
            prefix_thinks=prefix_thinks,
            prefix_step_count=keep_steps,
            total_step_count=total_steps,
            prefix_token_count=prefix_token_count,
            total_token_count=len(trajectory.full_prefix_token_ids),
            hint_boundaries=self._hint_boundaries(total_steps),
        )

    def build_prefix_plans(
        self,
        infos: list[dict[str, Any]],
        is_train: bool,
        group_size: int = 1,
        share_within_group: bool = True,
    ) -> list[StepHintPlan]:
        if not is_train:
            return [
                self._empty_plan(
                    trial_id=resolve_trial_id_from_gamefile(str(info.get("extra.gamefile") or "")),
                    gamefile=str(info.get("extra.gamefile") or ""),
                    slot_type="unhinted",
                    level=0,
                )
                for info in infos
            ]

        expected_group_size = self.expected_group_size()
        if int(group_size) != expected_group_size and bool(self.cfg.get("strict_group_size", True)):
            raise ValueError(
                f"StepHint strict reproduction expects env.rollout.n={expected_group_size}, "
                f"got {group_size}. Set +step_hint.strict_group_size=false to allow truncation."
            )

        plans: list[StepHintPlan] = []
        normalized_group_size = max(int(group_size), 1)
        for group_start in range(0, len(infos), normalized_group_size):
            group_infos = infos[group_start : group_start + normalized_group_size]
            source_info = group_infos[0]
            gamefile = str(source_info.get("extra.gamefile") or "")
            trial_id = resolve_trial_id_from_gamefile(gamefile)
            trajectory = self.store.get(trial_id)
            specs = self._slot_specs(trajectory)
            if len(specs) < len(group_infos):
                specs.extend([("unhinted", 0, 0)] * (len(group_infos) - len(specs)))
            for _, (slot_type, level, keep_steps) in zip(group_infos, specs):
                plans.append(
                    self._build_plan(
                        trajectory=trajectory,
                        trial_id=trial_id,
                        gamefile=gamefile,
                        slot_type=slot_type,
                        level=level,
                        keep_steps=keep_steps,
                    )
                )

        self.last_plans = plans
        return plans

    def build_batch_metadata(self, plans: list[StepHintPlan]) -> dict[str, np.ndarray]:
        return {
            "step_hint_trial_id": np.array([plan.trial_id for plan in plans], dtype=object),
            "step_hint_gamefile": np.array([plan.gamefile for plan in plans], dtype=object),
            "step_hint_matched": np.array([plan.matched for plan in plans], dtype=bool),
            "step_hint_task_type": np.array([plan.task_type for plan in plans], dtype=object),
            "step_hint_difficulty_group": np.array([plan.difficulty_group for plan in plans], dtype=np.int64),
            "step_hint_level": np.array([plan.level for plan in plans], dtype=np.int64),
            "step_hint_slot_type": np.array([plan.slot_type for plan in plans], dtype=object),
            "step_hint_prefix_step_count": np.array([plan.prefix_step_count for plan in plans], dtype=np.int64),
            "step_hint_total_step_count": np.array([plan.total_step_count for plan in plans], dtype=np.int64),
            "step_hint_prefix_token_count": np.array([plan.prefix_token_count for plan in plans], dtype=np.int64),
            "step_hint_total_token_count": np.array([plan.total_token_count for plan in plans], dtype=np.int64),
            "step_hint_is_reference": np.array([plan.is_reference for plan in plans], dtype=bool),
        }

    def update_after_train_batch(self, non_tensor_batch: dict[str, Any]) -> dict[str, float]:
        slot_types = np.asarray(non_tensor_batch.get("step_hint_slot_type", []), dtype=object)
        if slot_types.size == 0:
            return {}
        metrics: dict[str, float] = {}
        denom = max(float(slot_types.size), 1.0)
        for slot_type in sorted({str(value) for value in slot_types}):
            metrics[f"step_hint/train/{slot_type}_row_ratio"] = float(np.sum(slot_types == slot_type) / denom)

        protected = non_tensor_batch.get("step_hint_forced", None)
        if protected is not None:
            forced = np.asarray(protected, dtype=bool)
            metrics["step_hint/train/forced_row_ratio"] = float(forced.mean()) if forced.size else 0.0
        return metrics

    def save_to_checkpoint(self, checkpoint_folder: str) -> None:
        return None


class StepHintWebshopRuntime(StepHintAlfworldRuntime):
    """Strict StepHint plan builder for WebShop target-key matched tasks."""

    def _default_expert_json_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "sft_data" / "webshop_sft_data.json")

    def _load_expert_store(self) -> WebshopExpertTrajectoryStore:
        json_path = self._expert_json_path()
        raw_trajectories = load_webshop_trajectories(json_path)
        action_counts = [len(item.get("actions", [])) for item in raw_trajectories]
        difficulty_group_mode = str(self.cfg.get("difficulty_group_mode", "webshop_3"))
        if difficulty_group_mode in {"webshop_3", "webshop_three"}:
            group_thresholds = _compute_webshop_three_group_thresholds(action_counts)
        elif difficulty_group_mode == "short_medium_long":
            group_thresholds = _compute_short_medium_long_thresholds(action_counts)
        else:
            num_groups = int(self.cfg.get("num_difficulty_groups", 5))
            group_thresholds = _compute_group_thresholds(action_counts, num_groups=max(1, num_groups))

        temp_store = WebshopExpertTrajectoryStore({}, group_thresholds)
        trajectories: dict[Any, ExpertTrajectory] = {}
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
            if target_key in trajectories:
                continue

            step_end_offsets, full_prefix_token_ids = build_step_end_offsets(self.tokenizer, actions)
            action_count = len(actions)
            trial_id = str(item.get("id") or target_key_to_string(target_key))
            trajectories[target_key] = ExpertTrajectory(
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
            print(f"[StepHint WebShop] skipped {skipped} expert trajectories without target key")
        print(f"[StepHint WebShop] loaded {len(trajectories)} unique expert target keys from {json_path}")
        return WebshopExpertTrajectoryStore(trajectories, group_thresholds)

    def should_apply(self, env_name: str, is_train: bool) -> bool:
        if not bool(self.cfg.get("enable", False)):
            return False
        if "webshop" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.cfg.get("apply_on_train", True))
        return bool(self.cfg.get("apply_on_validation", False))

    def build_prefix_plans(
        self,
        infos: list[dict[str, Any]],
        is_train: bool,
        group_size: int = 1,
        share_within_group: bool = True,
    ) -> list[StepHintPlan]:
        if not is_train:
            return [
                self._empty_plan(
                    trial_id=target_key_to_string(normalize_target_key(info.get("webshop_target_key"))),
                    gamefile="",
                    slot_type="unhinted",
                    level=0,
                )
                for info in infos
            ]

        expected_group_size = self.expected_group_size()
        if int(group_size) != expected_group_size and bool(self.cfg.get("strict_group_size", True)):
            raise ValueError(
                f"StepHint strict reproduction expects env.rollout.n={expected_group_size}, "
                f"got {group_size}. Set +step_hint.strict_group_size=false to allow truncation."
            )

        plans: list[StepHintPlan] = []
        normalized_group_size = max(int(group_size), 1)
        for group_start in range(0, len(infos), normalized_group_size):
            group_infos = infos[group_start : group_start + normalized_group_size]
            target_key = normalize_target_key(group_infos[0].get("webshop_target_key"))
            if share_within_group:
                for info in group_infos[1:]:
                    group_target_key = normalize_target_key(info.get("webshop_target_key"))
                    if group_target_key != target_key:
                        raise ValueError(
                            "WebShop grouped rollouts must share one target key for StepHint; got "
                            f"{target_key_to_string(target_key)} and {target_key_to_string(group_target_key)}"
                        )

            trajectory = self.store.get(target_key)
            trial_id = trajectory.trial_id if trajectory is not None else target_key_to_string(target_key)
            specs = self._slot_specs(trajectory)
            if len(specs) < len(group_infos):
                specs.extend([("unhinted", 0, 0)] * (len(group_infos) - len(specs)))
            for _, (slot_type, level, keep_steps) in zip(group_infos, specs):
                plans.append(
                    self._build_plan(
                        trajectory=trajectory,
                        trial_id=trial_id,
                        gamefile="",
                        slot_type=slot_type,
                        level=level,
                        keep_steps=keep_steps,
                    )
                )

        self.last_plans = plans
        return plans


def build_stephint_runtime(tokenizer, config) -> StepHintAlfworldRuntime | StepHintWebshopRuntime | None:
    if not bool(config.get("step_hint", {}).get("enable", False)):
        return None
    env_name = str(config.env.env_name).lower()
    if "webshop" in env_name:
        return StepHintWebshopRuntime(tokenizer=tokenizer, config=config)
    if "alfworld" not in env_name:
        return None
    return StepHintAlfworldRuntime(tokenizer=tokenizer, config=config)
