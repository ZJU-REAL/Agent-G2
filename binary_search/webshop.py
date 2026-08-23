from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import ray

from agent_system.environments.base import to_numpy
from agent_system.environments.env_manager import WebshopEnvironmentManager
from agent_system.memory import SimpleMemory
from gmsv.alfworld import ExpertTrajectory, build_step_end_offsets
from gmsv.webshop import (
    WebshopExpertTrajectoryStore,
    _compute_webshop_three_group_thresholds,
    item_to_target_key,
    load_webshop_trajectories,
    normalize_target_key,
    target_key_to_string,
)

from .runtime import BinarySearchAlfworldPrefixRuntime, BinarySearchPrefixPlan


class FixedTargetWebshopEnvironmentManager(WebshopEnvironmentManager):
    def __init__(self, base_envs, indices: list[int]):
        self.base_envs = base_envs
        self.envs = base_envs.envs
        self.projection_f = base_envs.projection_f
        self.config = base_envs.config
        self.indices = [int(index) for index in indices]
        self.memory = SimpleMemory()
        self.prefix_terminal_infos = None
        self.prefix_sft_records = None
        self.prefix_detail_records = None
        self.last_infos = None
        self.tasks = []
        self.pre_text_obs = []

    def _target_key_from_env_kwarg(self, item: Any):
        if not isinstance(item, dict):
            return None
        return normalize_target_key(item.get("webshop_target_key")) or normalize_target_key(item.get("gamefile"))

    def reset(self, kwargs):
        if kwargs is None:
            raise RuntimeError("binary_search WebShop fixed-target rollout requires webshop_target_key env_kwargs")

        env_kwargs = list(kwargs)
        target_keys = [self._target_key_from_env_kwarg(item) for item in env_kwargs]
        if len(target_keys) != len(self.indices):
            raise ValueError(
                "binary_search WebShop fixed-target rollout expected one target key per selected worker, "
                f"got {len(target_keys)} keys for {len(self.indices)} workers"
            )
        missing = [idx for idx, target_key in enumerate(target_keys) if target_key is None]
        if missing:
            raise ValueError(f"binary_search WebShop fixed-target rollout got empty target keys at local rows {missing[:5]}")

        futures = [
            self.envs._workers[global_idx].reset_by_target_key.remote(target_key)
            for global_idx, target_key in zip(self.indices, target_keys)
        ]
        results = ray.get(futures)
        text_obs, infos = [], []
        for obs, info in results:
            text_obs.append(obs)
            infos.append(info)

        self.tasks = self.extract_task(text_obs)
        text_obs = self.format_obs(text_obs)
        self.last_infos = infos
        self.prefix_terminal_infos = [None for _ in range(len(infos))]
        self.prefix_sft_records = None
        self.prefix_detail_records = None
        self.pre_text_obs = text_obs
        self.memory.reset(batch_size=len(infos))
        return {
            "text": self.build_text_obs(text_obs, infos, init=True),
            "image": None,
            "anchor": text_obs.copy(),
        }, infos

    def step(self, text_actions):
        actions, valids = self.projection_f(text_actions)
        text_obs, rewards, dones, infos = self.envs.step_selected(self.indices, actions)
        text_obs = self.format_obs(text_obs)

        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        self.pre_text_obs = text_obs
        self.last_infos = infos
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        return {
            "text": self.build_text_obs(text_obs, infos),
            "image": None,
            "anchor": text_obs.copy(),
        }, to_numpy(rewards), to_numpy(dones), infos

    def replay_prefix_plans(self, prefix_plans, collect_sft: bool = False, collect_detail: bool = False):
        batch_size = len(self.pre_text_obs)
        prefix_rewards = np.zeros(batch_size, dtype=np.float32)
        prefix_lengths = np.zeros(batch_size, dtype=np.float32)
        prefix_dones = np.zeros(batch_size, dtype=bool)
        self.prefix_terminal_infos = [None for _ in range(batch_size)]
        self.prefix_sft_records = [[] for _ in range(batch_size)] if collect_sft else None
        self.prefix_detail_records = None

        max_prefix_steps = max((int(plan.prefix_step_count) for plan in prefix_plans), default=0)
        for step_idx in range(max_prefix_steps):
            selected_indices = []
            selected_actions = []
            selected_thinks = []
            for env_idx, plan in enumerate(prefix_plans):
                if prefix_dones[env_idx]:
                    continue
                if step_idx >= int(plan.prefix_step_count):
                    continue
                selected_indices.append(env_idx)
                selected_actions.append(plan.prefix_actions[step_idx])
                prefix_thinks = getattr(plan, "prefix_thinks", None) or []
                selected_thinks.append(prefix_thinks[step_idx] if step_idx < len(prefix_thinks) else "")

            if not selected_indices:
                continue

            if collect_sft:
                init_prompt = all(len(self.memory[env_idx]) == 0 for env_idx in range(batch_size))
                prompt_texts = self.build_text_obs(self.pre_text_obs, self.last_infos, init=init_prompt)
                for env_idx, expert_action, expert_think in zip(selected_indices, selected_actions, selected_thinks):
                    self.prefix_sft_records[env_idx].append({
                        "obs_text": prompt_texts[env_idx],
                        "action": expert_action,
                        "think": expert_think,
                        "step_idx": step_idx,
                    })

            previous_text_obs = [self.pre_text_obs[env_idx] for env_idx in selected_indices]
            global_indices = [self.indices[env_idx] for env_idx in selected_indices]
            text_obs, rewards, dones, infos = self.envs.step_selected(global_indices, selected_actions)
            text_obs = self.format_obs(text_obs)
            self.memory.store_selected({"text_obs": previous_text_obs, "action": selected_actions}, selected_indices)

            rewards = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(dones, dtype=bool)
            for local_idx, env_idx in enumerate(selected_indices):
                self.pre_text_obs[env_idx] = text_obs[local_idx]
                self.last_infos[env_idx] = infos[local_idx]
                prefix_rewards[env_idx] += float(rewards[local_idx])
                prefix_lengths[env_idx] += 1.0
                prefix_dones[env_idx] = prefix_dones[env_idx] or bool(dones[local_idx])
                self.prefix_terminal_infos[env_idx] = infos[local_idx]

        init_prompt = all(len(self.memory[env_idx]) == 0 for env_idx in range(batch_size))
        return {
            "text": self.build_text_obs(self.pre_text_obs, self.last_infos, init=init_prompt),
            "image": None,
            "anchor": self.pre_text_obs.copy(),
        }, prefix_rewards, prefix_lengths, prefix_dones


class BinarySearchWebshopPrefixRuntime(BinarySearchAlfworldPrefixRuntime):
    """WebShop prefix runtime used by online binary-search prefix selection."""

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
            print(f"[BinarySearch WebShop] skipped {skipped} expert trajectories without target key")
        print(f"[BinarySearch WebShop] loaded {len(store_items)} unique expert target keys from {json_path}")
        return WebshopExpertTrajectoryStore(store_items, group_thresholds)

    def should_apply(self, env_name: str, is_train: bool) -> bool:
        if not bool(self.binary_search_cfg.get("enable", False)):
            return False
        if "webshop" not in str(env_name).lower():
            return False
        if is_train:
            return bool(self.binary_search_cfg.get("apply_on_train", True))
        return bool(self.binary_search_cfg.get("apply_on_validation", False))

    def _empty_plan(
        self,
        trial_id: str,
        gamefile: str = "",
        matched: bool = False,
    ) -> BinarySearchPrefixPlan:
        plan = super()._empty_plan(trial_id=trial_id, gamefile=gamefile, matched=matched)
        plan.task_type = "webshop"
        return plan

    def _build_plan(
        self,
        trajectory: ExpertTrajectory | None,
        trial_id: str,
        gamefile: str,
        group_index: int | None = None,
    ) -> BinarySearchPrefixPlan:
        if trajectory is None and bool(self.binary_search_cfg.get("strict_expert_match", False)):
            raise KeyError(f"No expert trajectory found for WebShop target_key={gamefile or trial_id}")
        return super()._build_plan(
            trajectory=trajectory,
            trial_id=trial_id,
            gamefile=gamefile,
            group_index=group_index,
        )

    def build_prefix_plans(
        self,
        infos: list[dict[str, Any]],
        is_train: bool,
        group_size: int = 1,
        share_within_group: bool = True,
    ) -> list[BinarySearchPrefixPlan]:
        plans: list[BinarySearchPrefixPlan] = []
        if not share_within_group:
            for info in infos:
                target_key = normalize_target_key(info.get("webshop_target_key"))
                trajectory = self.store.get(target_key)
                trial_id = trajectory.trial_id if trajectory is not None else target_key_to_string(target_key)
                gamefile = target_key_to_string(target_key)
                plans.append(self._build_plan(trajectory=trajectory, trial_id=trial_id, gamefile=gamefile))
            self.last_plans = plans
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
                        "binary_search.share_prefix_within_group=true; got "
                        f"{target_key_to_string(target_key)} and {target_key_to_string(group_target_key)}"
                    )

            trajectory = self.store.get(target_key)
            trial_id = trajectory.trial_id if trajectory is not None else target_key_to_string(target_key)
            gamefile = target_key_to_string(target_key)
            shared_plan = self._build_plan(
                trajectory=trajectory,
                trial_id=trial_id,
                gamefile=gamefile,
                group_index=group_start // normalized_group_size,
            )
            for _ in group_infos:
                plans.append(replace(shared_plan))

        self.last_plans = plans
        return plans
