from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .alfworld import (
    ExpertTrajectory,
    GMSVAlfworldPrefixRuntime,
    PrefixPlan,
    _compute_group_thresholds,
    _compute_short_medium_long_thresholds,
    build_step_end_offsets,
)

TargetKey = tuple[str, tuple[tuple[str, int], ...]]

CONTROL_CLICKS = {
    "< prev",
    "back to search",
    "next >",
    "description",
    "features",
    "reviews",
}


def normalize_text(text: Any) -> str:
    return " ".join(str(text).lower().split())


def normalize_actions(actions: list[Any]) -> list[str]:
    normalized: list[str] = []
    for action in actions:
        matches = re.findall(
            r"(search\[[^\]]*]|\bclick\[[^\]]*])",
            str(action),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if matches:
            normalized.extend(match.strip() for match in matches)
        else:
            normalized.append(str(action).strip())
    return normalized


def parse_click_value(action: Any) -> Optional[str]:
    match = re.fullmatch(r"\s*click\[(.*)]\s*", str(action), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return normalize_text(match.group(1))


def _looks_like_asin(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9]{10}", value.upper())) and value.upper().startswith("B")


def target_key_to_string(key: TargetKey | None) -> str:
    if key is None:
        return ""
    asin, options = key
    option_text = "|".join(f"{value}#{count}" for value, count in options)
    return f"{asin}::{option_text}"


def normalize_target_key(key: Any) -> TargetKey | None:
    if key is None:
        return None
    if isinstance(key, tuple) and len(key) == 2:
        asin, options = key
        return (
            str(asin).upper(),
            tuple(sorted((normalize_text(value), int(count)) for value, count in options)),
        )
    if isinstance(key, list) and len(key) == 2:
        asin, options = key
        return (
            str(asin).upper(),
            tuple(sorted((normalize_text(value), int(count)) for value, count in options)),
        )
    if isinstance(key, str) and "::" in key:
        asin, option_text = key.split("::", 1)
        options = []
        if option_text:
            pending_parts = []
            for item in option_text.split("|"):
                pending_parts.append(item)
                if not re.search(r"#\d+\s*$", item):
                    continue
                item_text = "|".join(pending_parts)
                pending_parts = []
                if "#" not in item:
                    continue
                value, count = item_text.rsplit("#", 1)
                options.append((normalize_text(value), int(count)))
        return (asin.upper(), tuple(sorted(options)))
    return None


def actions_to_target_key(actions: list[Any], known_asins: set[str] | None = None) -> TargetKey | None:
    asin = None
    options: list[str] = []
    known_asins = known_asins or set()

    for action in normalize_actions(actions):
        clicked = parse_click_value(action)
        if clicked is None:
            continue
        if clicked == "buy now":
            break

        upper_clicked = clicked.upper()
        if upper_clicked in known_asins or _looks_like_asin(upper_clicked):
            asin = upper_clicked
            options = []
            continue

        if asin is not None and clicked not in CONTROL_CLICKS:
            options.append(clicked)

    if asin is None:
        return None
    return (asin, tuple(sorted(Counter(options).items())))


def goal_to_target_key(goal: dict[str, Any]) -> TargetKey:
    values = goal.get("goal_options") or {}
    if isinstance(values, dict):
        option_values = [normalize_text(value) for value in values.values()]
    else:
        option_values = [normalize_text(value) for value in values]
    return (
        str(goal.get("asin", "")).upper(),
        tuple(sorted(Counter(option_values).items())),
    )


def load_webshop_trajectories(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "trajectories" in payload:
        return payload["trajectories"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported WebShop expert trajectory format: {path}")


def item_to_target_key(item: dict[str, Any], known_asins: set[str] | None = None) -> TargetKey | None:
    for field in ("webshop_target_key", "webshop_target_key_text", "id"):
        key = normalize_target_key(item.get(field))
        if key is not None:
            return key
    return actions_to_target_key(item.get("actions", []), known_asins=known_asins)


def load_webshop_target_key_items(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        if "trajectories" in payload:
            return payload["trajectories"]
        if "ordered_samples" in payload:
            return payload["ordered_samples"]
        if "unique_goals" in payload:
            return payload["unique_goals"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported WebShop target-key source format: {path}")


def load_expert_target_keys(
    path: str | os.PathLike[str] | None,
    known_asins: set[str] | None = None,
) -> set[TargetKey]:
    if not path:
        return set()
    keys: set[TargetKey] = set()
    for item in load_webshop_target_key_items(path):
        key = item_to_target_key(item, known_asins=known_asins)
        if key is not None:
            keys.add(key)
    return keys


def load_webshop_target_key_sequence(
    path: str | os.PathLike[str] | None,
    known_asins: set[str] | None = None,
) -> list[TargetKey]:
    if not path:
        return []
    items = load_webshop_target_key_items(path)
    keys: list[TargetKey] = []
    for item in items:
        key = item_to_target_key(item, known_asins=known_asins)
        if key is not None:
            keys.append(key)
    return keys


def load_webshop_goal_idx_sequence(path: str | os.PathLike[str] | None) -> list[int]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "ordered_samples" not in payload:
        return []
    goal_idxs = []
    for item in payload["ordered_samples"]:
        if not isinstance(item, dict) or "goal_idx" not in item:
            return []
        goal_idxs.append(int(item["goal_idx"]))
    return goal_idxs


class WebshopExpertTrajectoryStore:
    def __init__(self, trajectories: dict[TargetKey, ExpertTrajectory], group_thresholds: list[int]):
        self.trajectories = trajectories
        self.group_thresholds = group_thresholds

    def get(self, target_key: Any) -> Optional[ExpertTrajectory]:
        key = normalize_target_key(target_key)
        if key is None:
            return None
        return self.trajectories.get(key)

    def difficulty_group_for_length(self, action_count: int) -> int:
        for idx, threshold in enumerate(self.group_thresholds, start=1):
            if action_count <= threshold:
                return idx
        return len(self.group_thresholds)


def _compute_webshop_three_group_thresholds(action_counts: list[int]) -> list[int]:
    max_length = max(action_counts) if action_counts else 6
    return [4, 5, max(6, int(max_length))]


class GMSVWebshopPrefixRuntime(GMSVAlfworldPrefixRuntime):
    def _default_expert_json_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "sft_data" / "webshop_sft_data.json")

    def _load_expert_store(self) -> WebshopExpertTrajectoryStore:
        json_path = self._expert_json_path()
        raw_trajectories = load_webshop_trajectories(json_path)
        action_counts = [len(item.get("actions", [])) for item in raw_trajectories]
        difficulty_group_mode = str(self.gmsv_cfg.get("difficulty_group_mode", "auto"))
        if difficulty_group_mode in {"webshop_3", "webshop_three"}:
            group_thresholds = _compute_webshop_three_group_thresholds(action_counts)
        elif difficulty_group_mode == "short_medium_long":
            group_thresholds = _compute_short_medium_long_thresholds(action_counts)
        else:
            num_groups = int(self.gmsv_cfg.get("num_difficulty_groups", 5))
            group_thresholds = _compute_group_thresholds(action_counts, num_groups=max(1, num_groups))

        temp_store = WebshopExpertTrajectoryStore({}, group_thresholds)
        store_items: dict[TargetKey, ExpertTrajectory] = {}
        skipped = 0

        for item in raw_trajectories:
            actions = []
            thinks = []
            raw_actions = item.get("actions", [])
            raw_thinks = item.get("think", [])
            has_thinks = isinstance(raw_thinks, list) and len(raw_thinks) > 0

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
            print(f"[GMSV WebShop] skipped {skipped} expert trajectories without target key")
        print(f"[GMSV WebShop] loaded {len(store_items)} unique expert target keys from {json_path}")
        return WebshopExpertTrajectoryStore(store_items, group_thresholds)

    def should_apply(self, env_name: str, is_train: bool) -> bool:
        if not bool(self.gmsv_cfg.get("enable", False)):
            return False
        if "webshop" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.gmsv_cfg.get("apply_on_train", True))
        return bool(self.gmsv_cfg.get("apply_on_validation", False))

    def _sample_plan(self, trajectory: Optional[ExpertTrajectory], is_train: bool, trial_id: str) -> PrefixPlan:
        if trajectory is None:
            if bool(self.gmsv_cfg.get("strict_expert_match", False)):
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
        return super()._sample_plan(trajectory=trajectory, is_train=is_train, trial_id=trial_id)

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
                        "gmsv.share_prefix_within_group=true; got "
                        f"{target_key_to_string(target_key)} and {target_key_to_string(group_target_key)}"
                    )
            trajectory = self.store.get(target_key)
            trial_id = trajectory.trial_id if trajectory is not None else target_key_to_string(target_key)
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
            "gmsv_extended_to_step_end": np.array([plan.extended_to_step_end for plan in plans], dtype=bool),
            "gmsv_fixed_no_prefix_train": np.array([plan.fixed_no_prefix_train for plan in plans], dtype=bool),
        }


def build_gmsv_webshop_runtime(tokenizer, config) -> Optional[GMSVWebshopPrefixRuntime]:
    if not bool(config.get("gmsv", {}).get("enable", False)):
        return None
    if "webshop" not in str(config.env.env_name).lower():
        return None
    return GMSVWebshopPrefixRuntime(tokenizer=tokenizer, config=config)
