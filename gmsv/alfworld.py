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


def _compute_short_medium_long_thresholds(action_counts: list[int]) -> list[int]:
    max_length = max(action_counts) if action_counts else 8
    return [4, 7, max(8, int(max_length))]


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


class GMSVAlfworldPrefixRuntime:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.gmsv_cfg = config.gmsv
        self.rng = np.random.default_rng(int(self.gmsv_cfg.get("seed", 1)))
        self.store = self._load_expert_store()
        self.group_stats = {
            group_id: GroupStats(
                accuracy_ema=float(self.gmsv_cfg.get("initial_group_accuracy", 0.0)),
                variance_ema=float(self.gmsv_cfg.get("initial_group_variance", 0.0)),
            )
            for group_id in range(1, len(self.store.group_thresholds) + 1)
        }
        self.mu_global = clip_value(
            float(self.gmsv_cfg.get("mu_global_init", 0.0)),
            float(self.gmsv_cfg.get("mu_global_min", 0.0)),
            float(self.gmsv_cfg.get("mu_global_max", 1.0)),
        )
        self.loaded_global_step = 0
        self.train_batches_completed = 0
        self._initialized = False

    def _default_expert_json_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "sft_data" / "alfworld_sft_data.json")

    def _expert_json_path(self) -> str:
        configured_path = self.gmsv_cfg.get("expert_json_path")
        return str(configured_path) if configured_path else self._default_expert_json_path()

    def _load_expert_store(self) -> ExpertTrajectoryStore:
        json_path = self._expert_json_path()
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        raw_trajectories = payload.get("trajectories", [])
        action_counts = [len(item.get("actions", [])) for item in raw_trajectories]
        difficulty_group_mode = str(self.gmsv_cfg.get("difficulty_group_mode", "auto"))
        if difficulty_group_mode == "short_medium_long":
            group_thresholds = _compute_short_medium_long_thresholds(action_counts)
        else:
            num_groups = int(self.gmsv_cfg.get("num_difficulty_groups", 5))
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
        if not bool(self.gmsv_cfg.get("enable", False)):
            return False
        if "alfworld" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.gmsv_cfg.get("apply_on_train", True))
        return bool(self.gmsv_cfg.get("apply_on_validation", False))

    def _compute_sigma(self, group_id: int) -> float:
        if str(self.gmsv_cfg.get("sigma_mode", "fixed_sigma")) == "fixed_sigma":
            return float(self.gmsv_cfg.get("fixed_sigma", 0.25))
        variance = self.group_stats[group_id].variance_ema
        return float(max(float(self.gmsv_cfg.get("sigma_min", 0.1)), float(self.gmsv_cfg.get("gamma", 1.0)) * variance))

    def _compute_mu(self, group_id: int) -> float:
        stats = self.group_stats[group_id]
        mu = self.mu_global + float(self.gmsv_cfg.get("lambda_value", 0.0)) * (0.5 - stats.accuracy_ema)
        return clip_value(mu, 0.0, float(self.gmsv_cfg.get("max_length", 1.0)))

    def _sample_prefix_ratio(self, mu: float, sigma: float) -> tuple[float, float]:
        sample_mode = str(self.gmsv_cfg.get("prefix_sample_mode", "gaussian"))
        max_length = float(self.gmsv_cfg.get("max_length", 1.0))

        if sample_mode == "gaussian":
            sampled_ratio = float(self.rng.normal(loc=mu, scale=max(sigma, 1e-8)))
            return sampled_ratio, clip_value(sampled_ratio, 0.0, max_length)

        if sample_mode == "uniform_fixed":
            env_name = str(self.config.env.env_name).lower()
            if "alfworld" not in env_name:
                raise ValueError("gmsv.prefix_sample_mode=uniform_fixed is currently supported only for ALFWorld")
            ratio_max = min(1.0, max(0.0, max_length))
            half_width = float(self.gmsv_cfg.get("uniform_half_width", 0.25))
            if half_width < 0.0:
                raise ValueError(f"gmsv.uniform_half_width must be non-negative, got {half_width}")
            center = clip_value(mu, 0.0, ratio_max)
            low = max(0.0, center - half_width)
            high = min(ratio_max, center + half_width)
            sampled_ratio = float(self.rng.uniform(low, high)) if high > low else float(low)
            return sampled_ratio, sampled_ratio

        raise ValueError(f"Unsupported gmsv.prefix_sample_mode: {sample_mode}")

    def _state_file_path(self, checkpoint_folder: str) -> str:
        return os.path.join(checkpoint_folder, "gmsv_runtime_state.json")

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
        if bool(self.gmsv_cfg.get("frozen_no_prefix_baseline", False)):
            return False

        state_path = self._state_file_path(checkpoint_folder)
        if not os.path.exists(state_path):
            return False

        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.loaded_global_step = int(state.get("loaded_global_step", self.loaded_global_step))
        self.train_batches_completed = int(state.get("train_batches_completed", 0))
        self.mu_global = clip_value(
            float(state.get("mu_global", self.mu_global)),
            float(self.gmsv_cfg.get("mu_global_min", 0.0)),
            float(self.gmsv_cfg.get("mu_global_max", 1.0)),
        )

        scoreboard_state = state.get("scoreboard", {})
        if isinstance(scoreboard_state, dict):
            for group_id, stats in scoreboard_state.items():
                group_idx = int(group_id)
                if group_idx not in self.group_stats:
                    self.group_stats[group_idx] = GroupStats(
                        accuracy_ema=float(self.gmsv_cfg.get("initial_group_accuracy", 0.0)),
                        variance_ema=float(self.gmsv_cfg.get("initial_group_variance", 0.0)),
                    )
                self.group_stats[group_idx] = GroupStats(
                    accuracy_ema=float(stats.get("accuracy_ema", self.group_stats[group_idx].accuracy_ema)),
                    variance_ema=float(stats.get("variance_ema", self.group_stats[group_idx].variance_ema)),
                )
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
            "mu_global": float(self.mu_global),
            "scoreboard": {
                str(group_id): {
                    "accuracy_ema": float(stats.accuracy_ema),
                    "variance_ema": float(stats.variance_ema),
                }
                for group_id, stats in self.group_stats.items()
            },
        }
        with open(self._state_file_path(checkpoint_folder), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _snapshot_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for group_id, stats in self.group_stats.items():
            metrics[f"scoreboard/group_{group_id}/A_k"] = float(stats.accuracy_ema)
            metrics[f"scoreboard/group_{group_id}/V_k"] = float(stats.variance_ema)
            metrics[f"scoreboard/group_{group_id}/sigma_k"] = float(self._compute_sigma(group_id))
        return metrics

    def _update_scoreboard(self, grouped_prompt_accuracies: dict[int, list[float]]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        alpha = float(self.gmsv_cfg.get("alpha", 0.2))

        for group_id, accuracies in grouped_prompt_accuracies.items():
            if not accuracies:
                continue
            values = np.asarray(accuracies, dtype=np.float32)
            batch_acc = float(values.mean())
            batch_var = float(values.var())
            stats = self.group_stats[group_id]
            stats.accuracy_ema = (1.0 - alpha) * stats.accuracy_ema + alpha * batch_acc
            stats.variance_ema = (1.0 - alpha) * stats.variance_ema + alpha * batch_var
            metrics[f"scoreboard/group_{group_id}/batch_acc"] = batch_acc
            metrics[f"scoreboard/group_{group_id}/batch_var"] = batch_var
            metrics[f"scoreboard/group_{group_id}/A_k"] = float(stats.accuracy_ema)
            metrics[f"scoreboard/group_{group_id}/V_k"] = float(stats.variance_ema)
        return metrics

    def _update_mu_global(self, batch_acc: float) -> float:
        if bool(self.gmsv_cfg.get("frozen_no_prefix_baseline", False)):
            self.mu_global = 0.0
            return self.mu_global

        if batch_acc < float(self.gmsv_cfg.get("batch_acc_low", 0.4)):
            self.mu_global += float(self.gmsv_cfg.get("global_step_delta", 0.1))
        elif batch_acc > float(self.gmsv_cfg.get("batch_acc_high", 0.6)):
            self.mu_global -= float(self.gmsv_cfg.get("global_step_delta", 0.1))

        self.mu_global = clip_value(
            self.mu_global,
            float(self.gmsv_cfg.get("mu_global_min", 0.0)),
            float(self.gmsv_cfg.get("mu_global_max", 1.0)),
        )
        return self.mu_global

    def update_after_train_batch(self, non_tensor_batch: dict[str, Any]) -> dict[str, float]:
        self.ensure_initialized()

        traj_uids = non_tensor_batch.get("traj_uid")
        rollout_ids = traj_uids if traj_uids is not None else non_tensor_batch.get("uid")
        if rollout_ids is None:
            return {}

        difficulty_groups = np.asarray(non_tensor_batch.get("gmsv_difficulty_group", []))
        if difficulty_groups.size == 0:
            return {}

        episode_rewards = np.asarray(non_tensor_batch.get("episode_rewards", np.zeros(len(rollout_ids), dtype=np.float32)), dtype=np.float32)
        expert_matched = np.asarray(non_tensor_batch.get("gmsv_expert_matched", np.ones(len(rollout_ids), dtype=bool)), dtype=bool)
        fixed_no_prefix_flags = np.asarray(non_tensor_batch.get("gmsv_fixed_no_prefix_train", np.zeros(len(rollout_ids), dtype=bool)), dtype=bool)

        grouped_rollout_flags: dict[Any, list[float]] = defaultdict(list)
        rollout_to_group: dict[Any, int] = {}
        rollout_fixed_no_prefix: dict[Any, bool] = {}
        rollout_expert_matched: dict[Any, bool] = {}

        for rollout_id, group_id, episode_reward, is_matched, fixed_no_prefix in zip(
            rollout_ids,
            difficulty_groups,
            episode_rewards,
            expert_matched,
            fixed_no_prefix_flags,
        ):
            rollout_key = str(rollout_id)
            grouped_rollout_flags[rollout_key].append(float(episode_reward > 0.0))
            rollout_to_group[rollout_key] = int(group_id)
            rollout_fixed_no_prefix[rollout_key] = rollout_fixed_no_prefix.get(rollout_key, False) or bool(fixed_no_prefix)
            rollout_expert_matched[rollout_key] = rollout_expert_matched.get(rollout_key, False) or bool(is_matched)

        grouped_rollout_accuracies: dict[int, list[float]] = defaultdict(list)
        scoreboard_grouped_rollout_accuracies: dict[int, list[float]] = defaultdict(list)
        rollout_accs: list[float] = []
        scoreboard_rollout_accs: list[float] = []
        fixed_no_prefix_rollout_ids: set[str] = set()
        matched_rollout_count = 0

        for rollout_id, flags in grouped_rollout_flags.items():
            if not rollout_expert_matched.get(rollout_id, False):
                continue

            matched_rollout_count += 1
            rollout_acc = float(max(flags)) if flags else 0.0
            group_id = rollout_to_group[rollout_id]
            rollout_accs.append(rollout_acc)
            grouped_rollout_accuracies[group_id].append(rollout_acc)

            if rollout_fixed_no_prefix.get(rollout_id, False):
                fixed_no_prefix_rollout_ids.add(rollout_id)
                continue

            scoreboard_rollout_accs.append(rollout_acc)
            scoreboard_grouped_rollout_accuracies[group_id].append(rollout_acc)

        batch_accuracy = float(np.mean(np.asarray(rollout_accs, dtype=np.float32))) if rollout_accs else 0.0
        scoreboard_batch_acc = float(np.mean(np.asarray(scoreboard_rollout_accs, dtype=np.float32))) if scoreboard_rollout_accs else 0.0

        metrics = {
            "train/matched_prompt_count": float(matched_rollout_count),
            "train/fixed_no_prefix_prompt_count": float(len(fixed_no_prefix_rollout_ids)),
            "train/batch_prompt_accuracy": batch_accuracy,
            "train/batch_prompt_accuracy_for_scoreboard": scoreboard_batch_acc,
            "train/mu_global_before_update": float(self.mu_global),
        }

        if bool(self.gmsv_cfg.get("frozen_no_prefix_baseline", False)):
            metrics["train/mu_global_after_update"] = float(self.mu_global)
            metrics.update(self._snapshot_metrics())
            self.train_batches_completed += 1
            return metrics

        metrics.update(self._update_scoreboard(scoreboard_grouped_rollout_accuracies))
        metrics["train/mu_global_after_update"] = float(self._update_mu_global(scoreboard_batch_acc))
        metrics.update(self._snapshot_metrics())
        self.train_batches_completed += 1
        return metrics

    def _sample_plan(self, trajectory: Optional[ExpertTrajectory], is_train: bool, trial_id: str) -> PrefixPlan:
        if trajectory is None:
            if bool(self.gmsv_cfg.get("strict_expert_match", False)):
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
        mu = self._compute_mu(group_id)
        sigma = self._compute_sigma(group_id)

        force_no_prefix = False
        fixed_no_prefix_train = False
        if bool(self.gmsv_cfg.get("frozen_no_prefix_baseline", False)):
            force_no_prefix = True
            fixed_no_prefix_train = True
        if is_train and float(self.gmsv_cfg.get("train_no_prefix_ratio", 0.0)) > 0.0:
            sampled_no_prefix = bool(self.rng.random() < float(self.gmsv_cfg.get("train_no_prefix_ratio", 0.0)))
            force_no_prefix = force_no_prefix or sampled_no_prefix
            fixed_no_prefix_train = fixed_no_prefix_train or sampled_no_prefix
        if bool(self.gmsv_cfg.get("no_prefix_when_mu_zero", False)) and mu <= 0.0:
            force_no_prefix = True

        if force_no_prefix or total_steps <= 0 or total_tokens <= 0:
            return PrefixPlan(
                trial_id=trajectory.trial_id,
                matched=True,
                task_type=trajectory.task_type,
                difficulty_group=group_id,
                mu=mu,
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

        sampled_ratio, clipped_ratio = self._sample_prefix_ratio(mu=mu, sigma=sigma)
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

        allow_full_prefix = bool(self.gmsv_cfg.get("allow_full_prefix", False))
        if allow_full_prefix and "alfworld" not in str(self.config.env.env_name).lower():
            raise ValueError("gmsv.allow_full_prefix=true is currently supported only for ALFWorld")

        if allow_full_prefix:
            max_prefix_steps = max(min(total_steps, int(self.config.env.max_steps)), 0)
        else:
            max_prefix_steps = max(
                min(
                    total_steps - 1,
                    int(self.config.env.max_steps) - 1,
                ),
                0,
            )
        if keep_steps > max_prefix_steps:
            keep_steps = max_prefix_steps
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
            mu=mu,
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
            "gmsv_trial_id": np.array([plan.trial_id for plan in plans], dtype=object),
            "gmsv_expert_matched": np.array([plan.matched for plan in plans], dtype=bool),
            "gmsv_task_type": np.array([plan.task_type for plan in plans], dtype=object),
            "gmsv_difficulty_group": np.array([plan.difficulty_group for plan in plans], dtype=np.int64),
            "gmsv_mu": np.array([plan.mu for plan in plans], dtype=np.float32),
            "gmsv_sigma": np.array([plan.sigma for plan in plans], dtype=np.float32),
            "gmsv_sampled_ratio": np.array([plan.sampled_ratio for plan in plans], dtype=np.float32),
            "gmsv_hint_ratio": np.array([plan.clipped_ratio for plan in plans], dtype=np.float32),
            "gmsv_prefix_step_count": np.array([plan.prefix_step_count for plan in plans], dtype=np.int64),
            "gmsv_prefix_token_count": np.array([plan.prefix_token_count for plan in plans], dtype=np.int64),
            "gmsv_total_step_count": np.array([plan.total_step_count for plan in plans], dtype=np.int64),
            "gmsv_total_token_count": np.array([plan.total_token_count for plan in plans], dtype=np.int64),
            "gmsv_full_prefix": np.array(
                [
                    plan.matched and plan.total_step_count > 0 and plan.prefix_step_count >= plan.total_step_count
                    for plan in plans
                ],
                dtype=bool,
            ),
            "gmsv_extended_to_step_end": np.array([plan.extended_to_step_end for plan in plans], dtype=bool),
            "gmsv_fixed_no_prefix_train": np.array([plan.fixed_no_prefix_train for plan in plans], dtype=bool),
        }


def build_gmsv_runtime(tokenizer, config) -> Optional[GMSVAlfworldPrefixRuntime]:
    if not bool(config.get("gmsv", {}).get("enable", False)):
        return None
    if "alfworld" not in str(config.env.env_name).lower():
        return None
    return GMSVAlfworldPrefixRuntime(tokenizer=tokenizer, config=config)
