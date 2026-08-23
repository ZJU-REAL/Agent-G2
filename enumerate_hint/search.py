from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ray

from verl import DataProto

from agent_system.environments.base import to_numpy
from agent_system.environments.env_manager import parse_gamefile, set_gamefile
from agent_system.environments.env_manager import WebshopEnvironmentManager
from agent_system.environments.prompts.alfworld import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS
from agent_system.memory import SimpleMemory
from gmsv.webshop import normalize_target_key

from .runtime import EnumerateHintAlfworldPrefixRuntime, EnumerateHintPrefixPlan


@dataclass
class CandidateResult:
    name: str
    output: DataProto
    repeated_gen_batch: DataProto
    prefix_sft_records: list[list[dict[str, Any]]] | None
    traj_uid: np.ndarray
    episode_rewards: np.ndarray
    plans: list[EnumerateHintPrefixPlan]
    success_count_by_group: dict[int, int]
    rollout_count_by_group: dict[int, int]
    group_to_local_index: dict[int, int]


@dataclass
class GroupEnumerateState:
    group_index: int
    gamefile: str
    trial_id: str
    total_steps: int
    selected_ratio: float | None = None
    selected_step: int | None = None
    selected_status: str | None = None
    selected_candidate: CandidateResult | None = None
    selected_success_count: int | None = None
    duplicate_skip_count: int = 0
    visited: dict[int, tuple[int, float, CandidateResult]] = field(default_factory=dict)


class _SelectedAlfworldEnvironmentManager:
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
        self.gamefile = []

    def _selected_admissible_commands(self):
        return [self.envs.get_admissible_commands[index] for index in self.indices]

    def reset(self, kwargs):
        if not hasattr(self.envs, "reset_selected"):
            raise RuntimeError("enumerate_hint subset rollout requires ALFWorld reset_selected support")
        text_obs, image_obs, infos = self.envs.reset_selected(self.indices, kwargs=kwargs)
        self.gamefile = parse_gamefile(infos)
        self.memory.reset(batch_size=len(text_obs))
        self.prefix_terminal_infos = [None for _ in range(len(text_obs))]
        self.prefix_sft_records = None
        self.prefix_detail_records = None
        self.last_infos = infos
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)
        full_text_obs = self.build_text_obs(text_obs, self._selected_admissible_commands(), init=True)
        return {"text": full_text_obs, "image": image_obs, "anchor": text_obs}, infos

    def step(self, text_actions):
        actions, valids = self.projection_f(text_actions, self._selected_admissible_commands())
        text_obs, image_obs, rewards, dones, infos = self.envs.step_selected(self.indices, actions)
        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        self.pre_text_obs = text_obs
        if infos and infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
        self.last_infos = infos
        full_text_obs = self.build_text_obs(text_obs, self._selected_admissible_commands())
        return {"text": full_text_obs, "image": image_obs, "anchor": text_obs}, to_numpy(rewards), to_numpy(dones), infos

    def replay_prefix_plans(self, prefix_plans, collect_sft: bool = False, collect_detail: bool = False):
        batch_size = len(self.pre_text_obs)
        prefix_rewards = np.zeros(batch_size, dtype=np.float32)
        prefix_lengths = np.zeros(batch_size, dtype=np.float32)
        prefix_dones = np.zeros(batch_size, dtype=bool)
        self.prefix_terminal_infos = [None for _ in range(batch_size)]
        self.prefix_sft_records = [[] for _ in range(batch_size)] if collect_sft else None
        self.prefix_detail_records = [[] for _ in range(batch_size)] if collect_detail else None

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

            prompt_texts = None
            if collect_sft or collect_detail:
                prompt_texts = self.build_text_obs(
                    self.pre_text_obs,
                    self._selected_admissible_commands(),
                    init=(step_idx == 0),
                )

            if collect_sft:
                for env_idx, expert_action, expert_think in zip(selected_indices, selected_actions, selected_thinks):
                    self.prefix_sft_records[env_idx].append({
                        "obs_text": prompt_texts[env_idx],
                        "action": expert_action,
                        "think": expert_think,
                        "step_idx": step_idx,
                    })

            previous_text_obs = [self.pre_text_obs[env_idx] for env_idx in selected_indices]
            global_indices = [self.indices[env_idx] for env_idx in selected_indices]
            text_obs, _, rewards, dones, infos = self.envs.step_selected(global_indices, selected_actions)
            self.memory.store_selected({"text_obs": previous_text_obs, "action": selected_actions}, selected_indices)

            if infos and infos[0].get("extra.gamefile") is None:
                infos = set_gamefile(infos, [self.gamefile[env_idx] for env_idx in selected_indices])

            rewards = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(dones, dtype=bool)
            for local_idx, env_idx in enumerate(selected_indices):
                self.pre_text_obs[env_idx] = text_obs[local_idx]
                self.last_infos[env_idx] = infos[local_idx]
                prefix_rewards[env_idx] += float(rewards[local_idx])
                prefix_lengths[env_idx] += 1.0
                prefix_dones[env_idx] = prefix_dones[env_idx] or bool(dones[local_idx])
                self.prefix_terminal_infos[env_idx] = infos[local_idx]

        full_text_obs = self.build_text_obs(self.pre_text_obs, self._selected_admissible_commands())
        image_obs = self.envs.getobs_selected(self.indices) if getattr(self.envs, "multi_modal", False) else None
        return {"text": full_text_obs, "image": image_obs, "anchor": self.pre_text_obs.copy()}, prefix_rewards, prefix_lengths, prefix_dones

    def extract_task(self, text_obs):
        for obs in text_obs:
            task_start = obs.find("Your task is to: ")
            if task_start != -1:
                self.tasks.append(obs[task_start + len("Your task is to: "):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def build_text_obs(self, text_obs, admissible_actions, init: bool = False):
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action",
            )

        for i in range(len(text_obs)):
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != "help")
            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                )
            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def success_evaluator(self, *args, **kwargs):
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        success = defaultdict(list)
        for batch_idx in range(len(total_batch_list)):
            self._process_batch(batch_idx, total_batch_list, total_infos, success)
        assert len(success["success_rate"]) == len(total_batch_list)
        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info["won"])
                success["success_rate"].append(won_value)
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return

        prefix_info = None
        if self.prefix_terminal_infos is not None:
            prefix_info = self.prefix_terminal_infos[batch_idx]
        if prefix_info is not None:
            won_value = float(prefix_info["won"])
            success["success_rate"].append(won_value)
            gamefile = prefix_info.get("extra.gamefile")
            if gamefile:
                self._process_gamefile(gamefile, won_value, success)

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class _SelectedWebshopEnvironmentManager(WebshopEnvironmentManager):
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
            raise RuntimeError("enumerate_hint WebShop subset rollout requires fixed webshop_target_key env_kwargs")

        env_kwargs = list(kwargs)
        target_keys = [self._target_key_from_env_kwarg(item) for item in env_kwargs]
        if len(target_keys) != len(self.indices):
            raise ValueError(
                "enumerate_hint WebShop subset rollout expected one target key per selected worker, "
                f"got {len(target_keys)} keys for {len(self.indices)} workers"
            )
        missing = [idx for idx, target_key in enumerate(target_keys) if target_key is None]
        if missing:
            raise ValueError(f"enumerate_hint WebShop subset rollout got empty target keys at local rows {missing[:5]}")

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


def _target_success_count(group_size: int, cfg) -> int:
    if cfg.get("target_success_count") is not None:
        return int(cfg.get("target_success_count"))
    target_rate = float(cfg.get("target_success_rate", 0.5))
    return int(math.ceil(target_rate * group_size))


def _target_success_rate(group_size: int, cfg) -> float:
    if cfg.get("target_success_rate") is not None:
        return float(cfg.get("target_success_rate"))
    if cfg.get("target_success_count") is not None:
        return float(cfg.get("target_success_count")) / max(float(group_size), 1.0)
    return 0.5


def _ratio_sequence(cfg) -> list[float]:
    configured = cfg.get("ratios", None)
    if configured is None:
        configured = cfg.get("ratio_sequence", None)
    if configured is None:
        return [0.0, 0.2, 0.4, 0.6, 0.8]
    ratios = [float(value) for value in configured]
    if not ratios:
        raise ValueError("enumerate_hint.ratios must not be empty")
    return ratios


def _make_gen_batch_with_gamefiles(gen_batch: DataProto, gamefiles: list[str] | None) -> DataProto:
    next_batch = gen_batch.select(deepcopy=True)
    if gamefiles is not None:
        next_batch.non_tensor_batch["env_kwargs"] = np.array(
            [{"gamefile": str(gamefile), "webshop_target_key": str(gamefile)} for gamefile in gamefiles],
            dtype=object,
        )
    return next_batch


def _remap_group_metadata(output: DataProto, local_to_group: dict[int, int]) -> None:
    group_values = np.asarray(output.non_tensor_batch.get("enumerate_hint_group_index", []), dtype=np.int64)
    if group_values.size == 0:
        return
    output.non_tensor_batch["enumerate_hint_group_index"] = np.array(
        [int(local_to_group.get(int(value), int(value))) for value in group_values],
        dtype=np.int64,
    )


def _run_candidate(
    *,
    collector,
    gen_batch: DataProto,
    actor_rollout_wg,
    envs,
    runtime: EnumerateHintAlfworldPrefixRuntime,
    group_size: int,
    candidate_name: str,
    group_indices: list[int],
    ratio_by_group: dict[int, float] | None = None,
    step_count_by_group: dict[int, int] | None = None,
    gamefiles: list[str] | None = None,
) -> CandidateResult:
    if not group_indices:
        raise ValueError("enumerate_hint candidate requires at least one group")
    if ratio_by_group is not None and step_count_by_group is not None:
        raise ValueError("enumerate_hint candidate accepts either ratio_by_group or step_count_by_group, not both")
    if ratio_by_group is None and step_count_by_group is None:
        ratio_by_group = {}

    local_to_group = {local_idx: int(group_idx) for local_idx, group_idx in enumerate(group_indices)}
    group_to_local = {int(group_idx): local_idx for local_idx, group_idx in local_to_group.items()}

    candidate_envs = envs
    if len(group_indices) != len(gen_batch):
        selected_worker_indices = [
            worker_idx
            for group_idx in group_indices
            for worker_idx in range(int(group_idx) * group_size, (int(group_idx) + 1) * group_size)
        ]
        if "webshop" in str(collector.config.env.env_name).lower():
            candidate_envs = _SelectedWebshopEnvironmentManager(envs, selected_worker_indices)
        else:
            candidate_envs = _SelectedAlfworldEnvironmentManager(envs, selected_worker_indices)

    candidate_gen_batch = gen_batch[group_indices]
    candidate_gamefiles = None
    if gamefiles is not None:
        candidate_gamefiles = [gamefiles[group_idx] for group_idx in group_indices]
    candidate_gen_batch = _make_gen_batch_with_gamefiles(candidate_gen_batch, candidate_gamefiles)
    repeated_gen_batch = candidate_gen_batch.repeat(repeat_times=group_size, interleave=True)

    if step_count_by_group is not None:
        local_steps = {
            group_to_local[int(group_idx)]: int(step_count)
            for group_idx, step_count in step_count_by_group.items()
            if int(group_idx) in group_to_local
        }
        runtime.configure_step_counts(default_step_count=0, step_count_by_group_index=local_steps)
    else:
        local_ratios = {
            group_to_local[int(group_idx)]: float(ratio)
            for group_idx, ratio in (ratio_by_group or {}).items()
            if int(group_idx) in group_to_local
        }
        runtime.configure_ratios(default_ratio=0.0, ratio_by_group_index=local_ratios)
    total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = collector.vanilla_multi_turn_loop(
        gen_batch=repeated_gen_batch,
        actor_rollout_wg=actor_rollout_wg,
        envs=candidate_envs,
        is_train=True,
        prefix_runtime_override=runtime,
        collect_rollout_details=False,
    )
    output = collector.gather_rollout_data(
        total_batch_list=total_batch_list,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        success=success,
        traj_uid=traj_uid,
        tool_callings=tool_callings,
    )
    _remap_group_metadata(output, local_to_group)

    success_count_by_group: dict[int, int] = {}
    rollout_count_by_group: dict[int, int] = {}
    for local_start in range(0, len(episode_rewards), group_size):
        local_idx = local_start // group_size
        group_idx = int(local_to_group[local_idx])
        group_rewards = episode_rewards[local_start : local_start + group_size]
        success_count_by_group[group_idx] = int(np.sum(np.asarray(group_rewards, dtype=np.float32) > 0.0))
        rollout_count_by_group[group_idx] = int(len(group_rewards))

    return CandidateResult(
        name=candidate_name,
        output=output,
        repeated_gen_batch=repeated_gen_batch,
        prefix_sft_records=getattr(candidate_envs, "prefix_sft_records", None),
        traj_uid=np.asarray(traj_uid, dtype=object),
        episode_rewards=np.asarray(episode_rewards, dtype=np.float32),
        plans=list(getattr(runtime, "last_plans", [])),
        success_count_by_group=success_count_by_group,
        rollout_count_by_group=rollout_count_by_group,
        group_to_local_index=group_to_local,
    )


def _candidate_group_plan(candidate: CandidateResult, group_idx: int, group_size: int) -> EnumerateHintPrefixPlan | None:
    local_idx = candidate.group_to_local_index.get(int(group_idx))
    if local_idx is None:
        return None
    plan_idx = local_idx * group_size
    if 0 <= plan_idx < len(candidate.plans):
        return candidate.plans[plan_idx]
    return None


def _select_group_rows(output: DataProto, group_idx: int) -> DataProto | None:
    group_values = np.asarray(output.non_tensor_batch.get("enumerate_hint_group_index", []), dtype=np.int64)
    if group_values.size == 0:
        return None
    indices = np.flatnonzero(group_values == int(group_idx)).astype(np.int64)
    if indices.size == 0:
        return None
    selected = output[indices]
    # Task-specific success_rate fields are aggregate logging metrics for the
    # candidate rollout. Different ALFWorld tasks expose different keys, so
    # keeping them makes DataProto.concat produce inconsistent non-tensor sizes.
    for key in list(selected.non_tensor_batch.keys()):
        if key != "success_rate" and str(key).endswith("_success_rate"):
            selected.non_tensor_batch.pop(key, None)
    return selected


def _add_selection_metadata(
    data: DataProto,
    *,
    ratio: float,
    step: int,
    success_count: int,
    target_success_count: int,
    status: str,
    eval_count: int,
    duplicate_skip_count: int,
) -> DataProto:
    size = len(data)
    data.non_tensor_batch["enumerate_hint_selected_ratio"] = np.full(size, float(ratio), dtype=np.float32)
    data.non_tensor_batch["enumerate_hint_selected_step_count"] = np.full(size, int(step), dtype=np.int64)
    data.non_tensor_batch["enumerate_hint_selected_success_count"] = np.full(size, int(success_count), dtype=np.int64)
    data.non_tensor_batch["enumerate_hint_target_success_count"] = np.full(size, int(target_success_count), dtype=np.int64)
    data.non_tensor_batch["enumerate_hint_status"] = np.array([status for _ in range(size)], dtype=object)
    data.non_tensor_batch["enumerate_hint_eval_count"] = np.full(size, int(eval_count), dtype=np.int64)
    data.non_tensor_batch["enumerate_hint_duplicate_skip_count"] = np.full(size, int(duplicate_skip_count), dtype=np.int64)
    return data


def _build_selected_sft_batch(
    *,
    collector,
    candidate: CandidateResult,
    group_idx: int,
    group_size: int,
) -> DataProto | None:
    if not collector._prefix_sft_enabled(is_train=True):
        return None
    local_idx = candidate.group_to_local_index.get(int(group_idx))
    if local_idx is None:
        return None
    start = local_idx * group_size
    stop = min(start + group_size, len(candidate.traj_uid))
    keep_traj_uids = {str(uid) for uid in candidate.traj_uid[start:stop]}
    return collector._build_prefix_sft_batch(
        gen_batch=candidate.repeated_gen_batch,
        prefix_sft_records=candidate.prefix_sft_records,
        traj_uid=candidate.traj_uid,
        keep_traj_uids=keep_traj_uids,
    )


def _record_candidate(
    states: dict[int, GroupEnumerateState],
    candidate: CandidateResult,
    step_by_group: dict[int, int],
    ratio_by_group: dict[int, float],
) -> None:
    for group_idx, step in step_by_group.items():
        success_count = candidate.success_count_by_group.get(group_idx, 0)
        states[group_idx].visited[int(step)] = (int(success_count), float(ratio_by_group[group_idx]), candidate)


def _record_step_candidate(
    states: dict[int, GroupEnumerateState],
    candidate: CandidateResult,
    step_by_group: dict[int, int],
    group_size: int,
) -> None:
    for group_idx, step in step_by_group.items():
        success_count = candidate.success_count_by_group.get(group_idx, 0)
        plan = _candidate_group_plan(candidate, group_idx, group_size)
        ratio = float(plan.clipped_ratio) if plan is not None else 0.0
        states[group_idx].visited[int(step)] = (int(success_count), ratio, candidate)


def _finalize_state(
    state: GroupEnumerateState,
    *,
    ratio: float,
    step: int,
    success_count: int,
    candidate: CandidateResult,
    status: str,
) -> None:
    state.selected_ratio = float(ratio)
    state.selected_step = int(step)
    state.selected_success_count = int(success_count)
    state.selected_candidate = candidate
    state.selected_status = status


def _best_exhaustive_step(
    state: GroupEnumerateState,
    *,
    group_size: int,
    target_rate: float,
    tie_break: str,
) -> tuple[int, int, float, CandidateResult]:
    if not state.visited:
        raise RuntimeError(f"enumerate_hint group {state.group_index} has no exhaustive candidates")

    normalized_tie_break = str(tie_break).lower()

    def rank(item: tuple[int, tuple[int, float, CandidateResult]]):
        step, (success_count, _, candidate) = item
        rollout_count = candidate.rollout_count_by_group.get(state.group_index, group_size)
        success_rate = float(success_count) / max(float(rollout_count), 1.0)
        distance = abs(success_rate - float(target_rate))
        if normalized_tie_break == "higher_step":
            return (distance, -int(step))
        if normalized_tie_break == "higher_success":
            return (distance, -int(success_count), int(step))
        if normalized_tie_break == "lower_success":
            return (distance, int(success_count), int(step))
        return (distance, int(step))

    best_step, (success_count, ratio, candidate) = min(state.visited.items(), key=rank)
    return int(best_step), int(success_count), float(ratio), candidate


def _run_ratio_early_stop_multi_turn_loop(
    *,
    collector,
    gen_batch: DataProto,
    actor_rollout_wg,
    envs,
    global_step: int | None = None,
    rollout_detail_dump_dir: str | None = None,
    collect_rollout_details: bool = False,
) -> DataProto:
    cfg = collector.config.get("enumerate_hint", {})
    if bool(collector.config.algorithm.filter_groups.get("enable", False)):
        raise ValueError("enumerate_hint currently requires algorithm.filter_groups.enable=false")

    runtime = getattr(collector, "enumerate_hint_runtime", None)
    if runtime is None:
        raise RuntimeError("enumerate_hint is enabled but TrajectoryCollector has no enumerate_hint_runtime")

    group_size = max(int(collector.config.env.rollout.n), 1)
    target_success_count = _target_success_count(group_size, cfg)
    ratios = _ratio_sequence(cfg)
    train_batch_size = len(gen_batch)

    first_ratio = float(ratios[0])
    first_group_indices = list(range(train_batch_size))
    first_ratio_by_group = {group_idx: first_ratio for group_idx in first_group_indices}
    first_candidate = _run_candidate(
        collector=collector,
        gen_batch=gen_batch,
        actor_rollout_wg=actor_rollout_wg,
        envs=envs,
        runtime=runtime,
        group_size=group_size,
        candidate_name=f"ratio_{first_ratio:g}",
        group_indices=first_group_indices,
        ratio_by_group=first_ratio_by_group,
        gamefiles=None,
    )

    gamefiles: list[str] = []
    states: dict[int, GroupEnumerateState] = {}
    first_step_by_group: dict[int, int] = {}
    for group_idx in range(train_batch_size):
        plan = _candidate_group_plan(first_candidate, group_idx, group_size)
        if plan is None:
            raise RuntimeError(f"enumerate_hint failed to read prefix plan for group {group_idx}")
        gamefiles.append(plan.gamefile)
        states[group_idx] = GroupEnumerateState(
            group_index=group_idx,
            gamefile=plan.gamefile,
            trial_id=plan.trial_id,
            total_steps=int(plan.total_step_count),
        )
        first_step_by_group[group_idx] = int(plan.prefix_step_count)

    _record_candidate(states, first_candidate, first_step_by_group, first_ratio_by_group)

    unresolved: set[int] = set(states.keys())
    for group_idx in list(unresolved):
        state = states[group_idx]
        step = first_step_by_group[group_idx]
        success_count, ratio, candidate = state.visited[step]
        if success_count >= target_success_count:
            _finalize_state(state, ratio=ratio, step=step, success_count=success_count, candidate=candidate, status="hit")
            unresolved.remove(group_idx)

    for ratio in ratios[1:]:
        if not unresolved:
            break

        step_by_group: dict[int, int] = {}
        ratio_by_group: dict[int, float] = {}
        groups_to_run: list[int] = []
        for group_idx in sorted(unresolved):
            state = states[group_idx]
            step = runtime.step_count_for_ratio(state.trial_id, state.gamefile, ratio)
            step_by_group[group_idx] = step
            ratio_by_group[group_idx] = float(ratio)
            if step in state.visited:
                state.duplicate_skip_count += 1
            else:
                groups_to_run.append(group_idx)

        if groups_to_run:
            candidate = _run_candidate(
                collector=collector,
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                runtime=runtime,
                group_size=group_size,
                candidate_name=f"ratio_{float(ratio):g}",
                group_indices=groups_to_run,
                ratio_by_group={group_idx: float(ratio) for group_idx in groups_to_run},
                gamefiles=gamefiles,
            )
            _record_candidate(
                states,
                candidate,
                {group_idx: step_by_group[group_idx] for group_idx in groups_to_run},
                {group_idx: float(ratio) for group_idx in groups_to_run},
            )

        for group_idx in list(unresolved):
            state = states[group_idx]
            step = step_by_group[group_idx]
            success_count, evaluated_ratio, candidate = state.visited[step]
            if success_count >= target_success_count:
                status = "duplicate_hit" if float(evaluated_ratio) != float(ratio) else "hit"
                _finalize_state(
                    state,
                    ratio=float(evaluated_ratio),
                    step=step,
                    success_count=success_count,
                    candidate=candidate,
                    status=status,
                )
                unresolved.remove(group_idx)

    fallback_ratio = float(ratios[-1])
    for group_idx in list(unresolved):
        state = states[group_idx]
        fallback_step = runtime.step_count_for_ratio(state.trial_id, state.gamefile, fallback_ratio)
        if fallback_step not in state.visited:
            candidate = _run_candidate(
                collector=collector,
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                runtime=runtime,
                group_size=group_size,
                candidate_name=f"ratio_{fallback_ratio:g}_fallback",
                group_indices=[group_idx],
                ratio_by_group={group_idx: fallback_ratio},
                gamefiles=gamefiles,
            )
            _record_candidate(states, candidate, {group_idx: fallback_step}, {group_idx: fallback_ratio})
        success_count, _, candidate = state.visited[fallback_step]
        _finalize_state(
            state,
            ratio=fallback_ratio,
            step=fallback_step,
            success_count=success_count,
            candidate=candidate,
            status="max_ratio_fallback",
        )
        unresolved.remove(group_idx)

    selected_outputs: list[DataProto] = []
    selected_sft_batches: list[DataProto] = []
    for group_idx in sorted(states):
        state = states[group_idx]
        if state.selected_candidate is None or state.selected_step is None or state.selected_success_count is None:
            raise RuntimeError(f"enumerate_hint group {group_idx} was not finalized")

        selected = _select_group_rows(state.selected_candidate.output, group_idx)
        if selected is None or len(selected) == 0:
            continue
        selected = _add_selection_metadata(
            selected,
            ratio=float(state.selected_ratio or 0.0),
            step=int(state.selected_step),
            success_count=int(state.selected_success_count),
            target_success_count=target_success_count,
            status=str(state.selected_status),
            eval_count=len(state.visited),
            duplicate_skip_count=state.duplicate_skip_count,
        )
        selected_outputs.append(selected)

        sft_batch = _build_selected_sft_batch(
            collector=collector,
            candidate=state.selected_candidate,
            group_idx=group_idx,
            group_size=group_size,
        )
        if sft_batch is not None and len(sft_batch) > 0:
            selected_sft_batches.append(sft_batch)

    if not selected_outputs:
        raise RuntimeError("enumerate_hint produced no selected rollout rows for training")

    output = DataProto.concat(selected_outputs)
    collector.latest_prefix_sft_batch = DataProto.concat(selected_sft_batches) if selected_sft_batches else None
    collector._pending_prefix_sft_batches = []
    collector._last_prefix_sft_records = None
    collector._last_prefix_sft_traj_uid = None
    collector._last_prefix_detail_records = None

    statuses = [str(states[idx].selected_status) for idx in sorted(states)]
    selected_ratios = [float(states[idx].selected_ratio or 0.0) for idx in sorted(states)]
    selected_steps = [int(states[idx].selected_step or 0) for idx in sorted(states)]
    selected_successes = [int(states[idx].selected_success_count or 0) for idx in sorted(states)]
    duplicate_skips = [int(states[idx].duplicate_skip_count) for idx in sorted(states)]
    runtime.last_metadata = {
        "global_step": int(global_step) if global_step is not None else None,
        "search_mode": "ratio_early_stop",
        "target_success_count": int(target_success_count),
        "group_size": int(group_size),
        "ratios": [float(ratio) for ratio in ratios],
        "num_groups": int(len(states)),
        "mean_selected_ratio": float(np.mean(selected_ratios)) if selected_ratios else 0.0,
        "mean_selected_step": float(np.mean(selected_steps)) if selected_steps else 0.0,
        "mean_selected_success_count": float(np.mean(selected_successes)) if selected_successes else 0.0,
        "mean_duplicate_skip_count": float(np.mean(duplicate_skips)) if duplicate_skips else 0.0,
        "statuses": {status: statuses.count(status) for status in sorted(set(statuses))},
    }
    return output


def _run_exhaustive_steps_multi_turn_loop(
    *,
    collector,
    gen_batch: DataProto,
    actor_rollout_wg,
    envs,
    global_step: int | None = None,
    rollout_detail_dump_dir: str | None = None,
    collect_rollout_details: bool = False,
) -> DataProto:
    cfg = collector.config.get("enumerate_hint", {})
    if bool(collector.config.algorithm.filter_groups.get("enable", False)):
        raise ValueError("enumerate_hint currently requires algorithm.filter_groups.enable=false")

    runtime = getattr(collector, "enumerate_hint_runtime", None)
    if runtime is None:
        raise RuntimeError("enumerate_hint is enabled but TrajectoryCollector has no enumerate_hint_runtime")

    group_size = max(int(collector.config.env.rollout.n), 1)
    target_success_count = _target_success_count(group_size, cfg)
    target_success_rate = _target_success_rate(group_size, cfg)
    train_batch_size = len(gen_batch)

    first_group_indices = list(range(train_batch_size))
    first_step_by_group = {group_idx: 0 for group_idx in first_group_indices}
    first_candidate = _run_candidate(
        collector=collector,
        gen_batch=gen_batch,
        actor_rollout_wg=actor_rollout_wg,
        envs=envs,
        runtime=runtime,
        group_size=group_size,
        candidate_name="step_0",
        group_indices=first_group_indices,
        step_count_by_group=first_step_by_group,
        gamefiles=None,
    )

    gamefiles: list[str] = []
    states: dict[int, GroupEnumerateState] = {}
    max_step_by_group: dict[int, int] = {}
    for group_idx in range(train_batch_size):
        plan = _candidate_group_plan(first_candidate, group_idx, group_size)
        if plan is None:
            raise RuntimeError(f"enumerate_hint failed to read prefix plan for group {group_idx}")
        gamefiles.append(plan.gamefile)
        total_steps = int(plan.total_step_count)
        max_prefix_step = min(max(total_steps - 1, 0), max(int(collector.config.env.max_steps) - 1, 0))
        states[group_idx] = GroupEnumerateState(
            group_index=group_idx,
            gamefile=plan.gamefile,
            trial_id=plan.trial_id,
            total_steps=total_steps,
        )
        max_step_by_group[group_idx] = max_prefix_step

    _record_step_candidate(states, first_candidate, first_step_by_group, group_size)

    max_step = max(max_step_by_group.values(), default=0)
    for step in range(1, max_step + 1):
        groups_to_run = [
            group_idx
            for group_idx in range(train_batch_size)
            if step <= max_step_by_group.get(group_idx, 0)
        ]
        if not groups_to_run:
            continue

        step_by_group = {group_idx: step for group_idx in groups_to_run}
        candidate = _run_candidate(
            collector=collector,
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            runtime=runtime,
            group_size=group_size,
            candidate_name=f"step_{step}",
            group_indices=groups_to_run,
            step_count_by_group=step_by_group,
            gamefiles=gamefiles,
        )
        _record_step_candidate(states, candidate, step_by_group, group_size)

    tie_break = str(cfg.get("exhaustive_tie_break", "lower_step"))
    for group_idx in sorted(states):
        state = states[group_idx]
        step, success_count, ratio, candidate = _best_exhaustive_step(
            state,
            group_size=group_size,
            target_rate=target_success_rate,
            tie_break=tie_break,
        )
        _finalize_state(
            state,
            ratio=ratio,
            step=step,
            success_count=success_count,
            candidate=candidate,
            status="closest_target",
        )

    selected_outputs: list[DataProto] = []
    selected_sft_batches: list[DataProto] = []
    for group_idx in sorted(states):
        state = states[group_idx]
        if state.selected_candidate is None or state.selected_step is None or state.selected_success_count is None:
            raise RuntimeError(f"enumerate_hint group {group_idx} was not finalized")

        selected = _select_group_rows(state.selected_candidate.output, group_idx)
        if selected is None or len(selected) == 0:
            continue
        selected = _add_selection_metadata(
            selected,
            ratio=float(state.selected_ratio or 0.0),
            step=int(state.selected_step),
            success_count=int(state.selected_success_count),
            target_success_count=target_success_count,
            status=str(state.selected_status),
            eval_count=len(state.visited),
            duplicate_skip_count=state.duplicate_skip_count,
        )
        selected_outputs.append(selected)

        sft_batch = _build_selected_sft_batch(
            collector=collector,
            candidate=state.selected_candidate,
            group_idx=group_idx,
            group_size=group_size,
        )
        if sft_batch is not None and len(sft_batch) > 0:
            selected_sft_batches.append(sft_batch)

    if not selected_outputs:
        raise RuntimeError("enumerate_hint produced no selected rollout rows for training")

    output = DataProto.concat(selected_outputs)
    collector.latest_prefix_sft_batch = DataProto.concat(selected_sft_batches) if selected_sft_batches else None
    collector._pending_prefix_sft_batches = []
    collector._last_prefix_sft_records = None
    collector._last_prefix_sft_traj_uid = None
    collector._last_prefix_detail_records = None

    statuses = [str(states[idx].selected_status) for idx in sorted(states)]
    selected_ratios = [float(states[idx].selected_ratio or 0.0) for idx in sorted(states)]
    selected_steps = [int(states[idx].selected_step or 0) for idx in sorted(states)]
    selected_successes = [int(states[idx].selected_success_count or 0) for idx in sorted(states)]
    eval_counts = [int(len(states[idx].visited)) for idx in sorted(states)]
    runtime.last_metadata = {
        "global_step": int(global_step) if global_step is not None else None,
        "search_mode": "exhaustive_steps",
        "target_success_rate": float(target_success_rate),
        "target_success_count": int(target_success_count),
        "group_size": int(group_size),
        "num_groups": int(len(states)),
        "max_exhaustive_step": int(max_step),
        "tie_break": tie_break,
        "mean_selected_ratio": float(np.mean(selected_ratios)) if selected_ratios else 0.0,
        "mean_selected_step": float(np.mean(selected_steps)) if selected_steps else 0.0,
        "mean_selected_success_count": float(np.mean(selected_successes)) if selected_successes else 0.0,
        "mean_eval_count": float(np.mean(eval_counts)) if eval_counts else 0.0,
        "statuses": {status: statuses.count(status) for status in sorted(set(statuses))},
    }
    return output


def run_enumerate_hint_multi_turn_loop(
    *,
    collector,
    gen_batch: DataProto,
    actor_rollout_wg,
    envs,
    global_step: int | None = None,
    rollout_detail_dump_dir: str | None = None,
    collect_rollout_details: bool = False,
) -> DataProto:
    cfg = collector.config.get("enumerate_hint", {})
    search_mode = str(cfg.get("search_mode", cfg.get("mode", "ratio_early_stop"))).lower()
    if search_mode in {"exhaustive_steps", "step_exhaustive", "native_enumerate"}:
        return _run_exhaustive_steps_multi_turn_loop(
            collector=collector,
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            global_step=global_step,
            rollout_detail_dump_dir=rollout_detail_dump_dir,
            collect_rollout_details=collect_rollout_details,
        )
    if search_mode in {"ratio_early_stop", "ratio", "early_stop"}:
        return _run_ratio_early_stop_multi_turn_loop(
            collector=collector,
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            global_step=global_step,
            rollout_detail_dump_dir=rollout_detail_dump_dir,
            collect_rollout_details=collect_rollout_details,
        )
    raise ValueError(f"Unknown enumerate_hint.search_mode: {search_mode}")
