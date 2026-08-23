from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path


def clip_value(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _trial_id_aliases(trial_id: str) -> list[str]:
    normalized = str(trial_id).strip()
    if not normalized:
        return []

    aliases = [normalized]
    marker = "_trial_"
    if marker in normalized:
        suffix = normalized[normalized.index(marker) + 1 :]
        if suffix and suffix not in aliases:
            aliases.append(suffix)
    return aliases


def resolve_trial_id_from_gamefile(gamefile: str | None) -> str:
    if not gamefile:
        return ""

    normalized = os.path.normpath(str(gamefile))
    candidates: list[str] = []
    parent = os.path.basename(os.path.dirname(normalized))
    if parent:
        candidates.append(parent)

    for part in reversed(normalized.split(os.sep)):
        if "_trial_" in part and part not in candidates:
            candidates.append(part)

    return candidates[0] if candidates else ""


def format_expert_prefix(actions: list[str]) -> str:
    if not actions:
        return ""
    return "\n".join(f"Step {step_idx}: {action}" for step_idx, action in enumerate(actions, start=1))


def build_step_end_offsets(tokenizer, actions: list[str]) -> tuple[list[int], list[int]]:
    if not actions:
        return [], []

    offsets: list[int] = []
    for step_idx in range(len(actions)):
        prefix_text = format_expert_prefix(actions[: step_idx + 1])
        offsets.append(len(tokenizer.encode(prefix_text, add_special_tokens=False)))

    prefix_token_ids = tokenizer.encode(format_expert_prefix(actions), add_special_tokens=False)
    return offsets, prefix_token_ids


def _compute_group_thresholds(action_counts: list[int], num_groups: int) -> list[int]:
    if not action_counts:
        return [1 for _ in range(max(1, num_groups))]

    unique_lengths, length_counts = np.unique(np.asarray(action_counts, dtype=np.int32), return_counts=True)
    unique_lengths = unique_lengths.tolist()
    length_counts = length_counts.tolist()

    if len(unique_lengths) < num_groups:
        quantiles = np.linspace(0.0, 1.0, num_groups)
        thresholds: list[int] = []
        for quantile in quantiles[1:]:
            thresholds.append(int(np.quantile(action_counts, quantile)))
        deduped: list[int] = []
        last_value = None
        for value in thresholds:
            if last_value is None or value > last_value:
                deduped.append(max(1, value))
                last_value = value
        while len(deduped) < num_groups:
            next_value = deduped[-1] if deduped else 1
            deduped.append(next_value)
        return deduped[:num_groups]

    total_count = int(sum(length_counts))
    target_size = total_count / float(num_groups)
    prefix_counts = np.cumsum(length_counts).tolist()

    best_boundaries: Optional[tuple[int, ...]] = None
    best_score: Optional[tuple[float, int]] = None
    for boundaries in combinations(range(1, len(unique_lengths)), num_groups - 1):
        segment_sizes = []
        start_idx = 0
        for boundary_idx in (*boundaries, len(unique_lengths)):
            left_prefix = prefix_counts[start_idx - 1] if start_idx > 0 else 0
            right_prefix = prefix_counts[boundary_idx - 1]
            segment_sizes.append(int(right_prefix - left_prefix))
            start_idx = boundary_idx
        if any(size <= 0 for size in segment_sizes):
            continue

        spread = int(max(segment_sizes) - min(segment_sizes))
        imbalance = float(sum(abs(size - target_size) for size in segment_sizes))
        score = (spread, imbalance)
        if best_score is None or score < best_score:
            best_score = score
            best_boundaries = boundaries

    if best_boundaries is None:
        return [max(1, int(unique_lengths[-1]))] * num_groups

    thresholds = [max(1, int(unique_lengths[boundary_idx - 1])) for boundary_idx in best_boundaries]
    thresholds.append(max(1, int(unique_lengths[-1])))
    return thresholds


@dataclass
class GroupStats:
    accuracy_ema: float
    variance_ema: float


@dataclass
class ExpertTrajectory:
    trial_id: str
    task_type: str
    actions: list[str]
    thinks: list[str]
    action_count: int
    difficulty_group: int
    step_end_offsets: list[int]
    full_prefix_token_ids: list[int]


@dataclass
class PrefixPlan:
    trial_id: str
    matched: bool
    task_type: str
    difficulty_group: int
    mu: float
    sigma: float
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


class ExpertTrajectoryStore:
    def __init__(self, trajectories: dict[str, ExpertTrajectory], group_thresholds: list[int]):
        self.trajectories = trajectories
        self.group_thresholds = group_thresholds
        self.alias_to_trial_id: dict[str, str] = {}
        for canonical_trial_id in trajectories:
            for alias in _trial_id_aliases(canonical_trial_id):
                self.alias_to_trial_id.setdefault(alias, canonical_trial_id)

    def _resolve_trial_id(self, trial_id: str) -> Optional[str]:
        normalized = str(trial_id).strip()
        if not normalized:
            return None
        if normalized in self.trajectories:
            return normalized
        return self.alias_to_trial_id.get(normalized)

    def get(self, trial_id: str) -> Optional[ExpertTrajectory]:
        resolved_trial_id = self._resolve_trial_id(trial_id)
        if resolved_trial_id is None:
            return None
        return self.trajectories.get(resolved_trial_id)

    def difficulty_group_for_length(self, action_count: int) -> int:
        for idx, threshold in enumerate(self.group_thresholds, start=1):
            if action_count <= threshold:
                return idx
        return len(self.group_thresholds)


class LinerHintAlfworldPrefixRuntime:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.liner_hint_cfg = config.liner_hint
        self.rng = np.random.default_rng(int(self.liner_hint_cfg.get("seed", 1)))
        self.store = self._load_expert_store()
        self.loaded_global_step = 0
        self.train_batches_completed = 0
        self._initialized = False

    def _default_expert_json_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "sft_data" / "alfworld_sft_data.json")

    def _expert_json_path(self) -> str:
        configured_path = self.liner_hint_cfg.get("expert_json_path")
        return str(configured_path) if configured_path else self._default_expert_json_path()

    def _load_expert_store(self) -> ExpertTrajectoryStore:
        json_path = self._expert_json_path()
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        raw_trajectories = payload.get("trajectories", [])
        num_groups = int(self.liner_hint_cfg.get("num_difficulty_groups", 5))
        action_counts = [len(item.get("actions", [])) for item in raw_trajectories]
        group_thresholds = _compute_group_thresholds(action_counts, num_groups=max(1, num_groups))
        temp_store = ExpertTrajectoryStore({}, group_thresholds)
        store_items: dict[str, ExpertTrajectory] = {}

        for item in raw_trajectories:
            trial_id = str(item["id"])
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

            step_end_offsets, full_prefix_token_ids = build_step_end_offsets(self.tokenizer, actions)
            action_count = len(actions)
            difficulty_group = temp_store.difficulty_group_for_length(action_count)
            store_items[trial_id] = ExpertTrajectory(
                trial_id=trial_id,
                task_type=str(item.get("task_type", "")),
                actions=actions,
                thinks=thinks,
                action_count=action_count,
                difficulty_group=difficulty_group,
                step_end_offsets=step_end_offsets,
                full_prefix_token_ids=full_prefix_token_ids,
            )

        return ExpertTrajectoryStore(store_items, group_thresholds)

    def should_apply(self, env_name: str, is_train: bool) -> bool:
        if not bool(self.liner_hint_cfg.get("enable", False)):
            return False
        if "alfworld" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.liner_hint_cfg.get("apply_on_train", True))
        return bool(self.liner_hint_cfg.get("apply_on_validation", False))

    def _current_train_step(self) -> int:
        return int(self.loaded_global_step + self.train_batches_completed)

    def _linear_prefix_ratio(self, is_train: bool) -> float:
        if not is_train:
            return clip_value(
                float(self.liner_hint_cfg.get("validation_ratio", self.liner_hint_cfg.get("start_ratio", 1.0))),
                0.0,
                1.0,
            )

        start_ratio = float(self.liner_hint_cfg.get("start_ratio", 1.0))
        end_ratio = float(self.liner_hint_cfg.get("end_ratio", 0.0))
        decay_steps = int(self.liner_hint_cfg.get("decay_steps", self.config.trainer.get("total_epochs", 300)))
        decay_steps = max(decay_steps, 1)
        if bool(self.liner_hint_cfg.get("include_last_step", True)):
            denominator = max(decay_steps - 1, 1)
        else:
            denominator = decay_steps
        progress = clip_value(float(self._current_train_step()) / float(denominator), 0.0, 1.0)
        ratio = start_ratio + (end_ratio - start_ratio) * progress
        return clip_value(ratio, 0.0, float(self.liner_hint_cfg.get("max_length", 1.0)))

    def _state_file_path(self, checkpoint_folder: str) -> str:
        return os.path.join(checkpoint_folder, "liner_hint_runtime_state.json")

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

        difficulty_groups = np.asarray(non_tensor_batch.get("liner_hint_difficulty_group", []))
        if difficulty_groups.size == 0:
            return {}

        episode_rewards = np.asarray(non_tensor_batch.get("episode_rewards", np.zeros(len(rollout_ids), dtype=np.float32)), dtype=np.float32)
        expert_matched = np.asarray(non_tensor_batch.get("liner_hint_expert_matched", np.ones(len(rollout_ids), dtype=bool)), dtype=bool)

        grouped_rollout_flags: dict[str, list[float]] = defaultdict(list)
        rollout_expert_matched: dict[str, bool] = {}
        for rollout_id, episode_reward, is_matched in zip(rollout_ids, episode_rewards, expert_matched):
            rollout_key = str(rollout_id)
            grouped_rollout_flags[rollout_key].append(float(episode_reward > 0.0))
            rollout_expert_matched[rollout_key] = rollout_expert_matched.get(rollout_key, False) or bool(is_matched)

        rollout_accs: list[float] = []
        matched_rollout_count = 0
        for rollout_id, flags in grouped_rollout_flags.items():
            if not rollout_expert_matched.get(rollout_id, False):
                continue
            matched_rollout_count += 1
            rollout_accs.append(float(max(flags)) if flags else 0.0)

        batch_accuracy = float(np.mean(np.asarray(rollout_accs, dtype=np.float32))) if rollout_accs else 0.0
        metrics = {
            "liner_hint/train_step": float(self._current_train_step()),
            "liner_hint/current_ratio": float(self._linear_prefix_ratio(is_train=True)),
            "liner_hint/matched_prompt_count": float(matched_rollout_count),
            "liner_hint/batch_prompt_accuracy": batch_accuracy,
        }
        self.train_batches_completed += 1
        return metrics

    def _sample_plan(self, trajectory: Optional[ExpertTrajectory], is_train: bool, trial_id: str) -> PrefixPlan:
        if trajectory is None:
            if bool(self.liner_hint_cfg.get("strict_expert_match", False)):
                raise KeyError(f"No expert trajectory found for ALFWorld trial_id={trial_id}")
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

        total_steps = trajectory.action_count
        total_tokens = len(trajectory.full_prefix_token_ids)
        group_id = trajectory.difficulty_group
        clipped_ratio = self._linear_prefix_ratio(is_train=is_train)
        sigma = 0.0

        force_no_prefix = False
        fixed_no_prefix_train = False
        if bool(self.liner_hint_cfg.get("frozen_no_prefix_baseline", False)):
            force_no_prefix = True
            fixed_no_prefix_train = True
        if is_train and float(self.liner_hint_cfg.get("train_no_prefix_ratio", 0.0)) > 0.0:
            sampled_no_prefix = bool(self.rng.random() < float(self.liner_hint_cfg.get("train_no_prefix_ratio", 0.0)))
            force_no_prefix = force_no_prefix or sampled_no_prefix
            fixed_no_prefix_train = fixed_no_prefix_train or sampled_no_prefix
        if bool(self.liner_hint_cfg.get("no_prefix_when_mu_zero", True)) and clipped_ratio <= 0.0:
            force_no_prefix = True

        if force_no_prefix or total_steps <= 0 or total_tokens <= 0:
            return PrefixPlan(
                trial_id=trajectory.trial_id,
                matched=True,
                task_type=trajectory.task_type,
                difficulty_group=group_id,
                mu=clipped_ratio,
                sigma=sigma,
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

        sampled_ratio = clipped_ratio
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

        reserve_model_step = bool(self.liner_hint_cfg.get("reserve_model_step", True))
        if reserve_model_step and keep_steps >= total_steps:
            keep_steps = max(total_steps - 1, 0)
            extended = False

        max_budget_steps = max(int(self.config.env.max_steps), 0)
        if reserve_model_step and max_budget_steps > 0:
            max_budget_steps = max(max_budget_steps - 1, 0)
        if keep_steps > max_budget_steps:
            keep_steps = max_budget_steps
            extended = False

        prefix_actions = trajectory.actions[:keep_steps]
        prefix_thinks = trajectory.thinks[:keep_steps] if trajectory.thinks else []
        prefix_token_count = 0
        if prefix_actions:
            prefix_token_count = len(self.tokenizer.encode(format_expert_prefix(prefix_actions), add_special_tokens=False))

        return PrefixPlan(
            trial_id=trajectory.trial_id,
            matched=True,
            task_type=trajectory.task_type,
            difficulty_group=group_id,
            mu=clipped_ratio,
            sigma=sigma,
            sampled_ratio=sampled_ratio,
            clipped_ratio=clipped_ratio,
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
        return {
            "liner_hint_trial_id": np.array([plan.trial_id for plan in plans], dtype=object),
            "liner_hint_expert_matched": np.array([plan.matched for plan in plans], dtype=bool),
            "liner_hint_task_type": np.array([plan.task_type for plan in plans], dtype=object),
            "liner_hint_difficulty_group": np.array([plan.difficulty_group for plan in plans], dtype=np.int64),
            "liner_hint_mu": np.array([plan.mu for plan in plans], dtype=np.float32),
            "liner_hint_sigma": np.array([plan.sigma for plan in plans], dtype=np.float32),
            "liner_hint_sampled_ratio": np.array([plan.sampled_ratio for plan in plans], dtype=np.float32),
            "liner_hint_hint_ratio": np.array([plan.clipped_ratio for plan in plans], dtype=np.float32),
            "liner_hint_prefix_step_count": np.array([plan.prefix_step_count for plan in plans], dtype=np.int64),
            "liner_hint_prefix_token_count": np.array([plan.prefix_token_count for plan in plans], dtype=np.int64),
            "liner_hint_total_step_count": np.array([plan.total_step_count for plan in plans], dtype=np.int64),
            "liner_hint_total_token_count": np.array([plan.total_token_count for plan in plans], dtype=np.int64),
            "liner_hint_extended_to_step_end": np.array([plan.extended_to_step_end for plan in plans], dtype=bool),
            "liner_hint_fixed_no_prefix_train": np.array([plan.fixed_no_prefix_train for plan in plans], dtype=bool),
        }


def build_liner_hint_runtime(tokenizer, config) -> Optional[LinerHintAlfworldPrefixRuntime]:
    if not bool(config.get("liner_hint", {}).get("enable", False)):
        return None
    if "alfworld" not in str(config.env.env_name).lower():
        return None
    return LinerHintAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
