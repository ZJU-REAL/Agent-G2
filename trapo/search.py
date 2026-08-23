from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray

from agent_system.environments.base import to_numpy
from agent_system.environments.env_manager import AlfWorldEnvironmentManager, WebshopEnvironmentManager, parse_gamefile
from agent_system.memory import SimpleMemory
from gmsv.webshop import normalize_target_key
from verl import DataProto

from .runtime import TRAPOAlfworldPrefixRuntime


def _unpack_selected_step(step_result):
    if len(step_result) == 5:
        return step_result
    if len(step_result) == 4:
        text_obs, rewards, dones, infos = step_result
        return text_obs, None, rewards, dones, infos
    raise ValueError(f"Unexpected step_selected return size: {len(step_result)}")


class FixedTargetAlfworldEnvironmentManager(AlfWorldEnvironmentManager):
    """A view over selected ALFWorld workers.

    TRAPO runs micro-groups of size 4, 2, 1, and 1 while the base ALFWorld
    manager owns train_batch_size * rollout.n workers. This wrapper lets one
    micro-group use only the first n_i workers inside each rollout group.
    """

    def __init__(self, base_envs: AlfWorldEnvironmentManager, indices: list[int]):
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
        commands = self.envs.get_admissible_commands
        return [commands[index] for index in self.indices]

    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset_selected(self.indices, kwargs=kwargs)
        self.gamefile = parse_gamefile(infos)
        self.memory.reset(batch_size=len(text_obs))
        self.prefix_terminal_infos = [None for _ in range(len(text_obs))]
        self.prefix_sft_records = None
        self.prefix_detail_records = None
        self.tasks = []
        self.pre_text_obs = text_obs
        self.last_infos = infos
        self.extract_task(text_obs)
        return {
            "text": self.build_text_obs(text_obs, self._selected_admissible_commands(), init=True),
            "image": image_obs,
            "anchor": text_obs,
        }, infos

    def step(self, text_actions):
        actions, valids = self.projection_f(text_actions, self._selected_admissible_commands())
        text_obs, image_obs, rewards, dones, infos = _unpack_selected_step(
            self.envs.step_selected(self.indices, actions)
        )
        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        self.pre_text_obs = text_obs
        self.last_infos = infos

        if infos and infos[0].get("extra.gamefile") is None:
            for local_idx, info in enumerate(infos):
                info["extra.gamefile"] = self.gamefile[local_idx] if local_idx < len(self.gamefile) else None

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        return {
            "text": self.build_text_obs(text_obs, self._selected_admissible_commands()),
            "image": image_obs,
            "anchor": text_obs,
        }, to_numpy(rewards), to_numpy(dones), infos

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
            selected_local_indices = []
            selected_actions = []
            selected_thinks = []
            for env_idx, plan in enumerate(prefix_plans):
                if prefix_dones[env_idx] or step_idx >= int(plan.prefix_step_count):
                    continue
                selected_local_indices.append(env_idx)
                selected_actions.append(plan.prefix_actions[step_idx])
                prefix_thinks = getattr(plan, "prefix_thinks", None) or []
                selected_thinks.append(prefix_thinks[step_idx] if step_idx < len(prefix_thinks) else "")

            if not selected_local_indices:
                continue

            prompt_texts = None
            available_actions_snapshot = None
            if collect_sft or collect_detail:
                prompt_texts = self.build_text_obs(
                    self.pre_text_obs,
                    self._selected_admissible_commands(),
                    init=(step_idx == 0),
                )
            if collect_detail:
                available_actions_snapshot = self._selected_admissible_commands()

            if collect_sft:
                for env_idx, expert_action, expert_think in zip(selected_local_indices, selected_actions, selected_thinks):
                    self.prefix_sft_records[env_idx].append(
                        {
                            "obs_text": prompt_texts[env_idx],
                            "action": expert_action,
                            "think": expert_think,
                            "step_idx": step_idx,
                        }
                    )

            previous_text_obs = [self.pre_text_obs[env_idx] for env_idx in selected_local_indices]
            selected_global_indices = [self.indices[env_idx] for env_idx in selected_local_indices]
            text_obs, _, rewards, dones, infos = _unpack_selected_step(
                self.envs.step_selected(selected_global_indices, selected_actions)
            )
            self.memory.store_selected({"text_obs": previous_text_obs, "action": selected_actions}, selected_local_indices)

            for local_idx, env_idx in enumerate(selected_local_indices):
                if isinstance(infos[local_idx], dict) and infos[local_idx].get("extra.gamefile") is None:
                    infos[local_idx]["extra.gamefile"] = self.gamefile[env_idx] if env_idx < len(self.gamefile) else None
                reward_value = float(rewards[local_idx])
                next_cumulative_reward = float(prefix_rewards[env_idx] + reward_value)
                self.pre_text_obs[env_idx] = text_obs[local_idx]
                if self.last_infos is not None and env_idx < len(self.last_infos):
                    self.last_infos[env_idx] = infos[local_idx]
                prefix_rewards[env_idx] += reward_value
                prefix_lengths[env_idx] += 1.0
                prefix_dones[env_idx] = prefix_dones[env_idx] or bool(dones[local_idx])
                self.prefix_terminal_infos[env_idx] = infos[local_idx]
                if collect_detail:
                    plan = prefix_plans[env_idx]
                    available_actions = available_actions_snapshot[env_idx] if available_actions_snapshot is not None else []
                    self.prefix_detail_records[env_idx].append(
                        {
                            "step_id": int(step_idx),
                            "phase": "prefix",
                            "source": "expert",
                            "trial_id": str(getattr(plan, "trial_id", "")),
                            "prefix_step_count": int(getattr(plan, "prefix_step_count", 0)),
                            "total_step_count": int(getattr(plan, "total_step_count", 0)),
                            "prompt": prompt_texts[env_idx] if prompt_texts is not None else None,
                            "observation": previous_text_obs[local_idx],
                            "anchor_observation": previous_text_obs[local_idx],
                            "available_actions": available_actions,
                            "raw_model_output": None,
                            "think": selected_thinks[local_idx],
                            "action": selected_actions[local_idx],
                            "is_action_valid": selected_actions[local_idx] in available_actions,
                            "env_reward": reward_value,
                            "cumulative_reward": next_cumulative_reward,
                            "done": bool(dones[local_idx]),
                            "won": bool(infos[local_idx].get("won", False)) if isinstance(infos[local_idx], dict) else False,
                            "task_score": infos[local_idx].get("task_score") if isinstance(infos[local_idx], dict) else None,
                            "info": infos[local_idx],
                            "next_observation": text_obs[local_idx],
                        }
                    )

        return {
            "text": self.build_text_obs(self.pre_text_obs, self._selected_admissible_commands()),
            "image": self.envs.getobs_selected(self.indices) if getattr(self.envs, "multi_modal", False) else None,
            "anchor": self.pre_text_obs.copy(),
        }, prefix_rewards, prefix_lengths, prefix_dones


class FixedTargetWebshopEnvironmentManager(WebshopEnvironmentManager):
    """A selected-worker WebShop view for TRAPO micro-groups."""

    def __init__(self, base_envs: WebshopEnvironmentManager, indices: list[int]):
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
            all_text_obs, all_infos = self.envs.reset()
            text_obs = [all_text_obs[index] for index in self.indices]
            infos = [all_infos[index] for index in self.indices]
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

        env_kwargs = list(kwargs)
        target_keys = [self._target_key_from_env_kwarg(item) for item in env_kwargs]
        if len(target_keys) != len(self.indices):
            raise ValueError(
                "TRAPO WebShop subset rollout expected one target key per selected worker, "
                f"got {len(target_keys)} keys for {len(self.indices)} workers"
            )
        missing = [idx for idx, target_key in enumerate(target_keys) if target_key is None]
        if missing:
            raise ValueError(f"TRAPO WebShop subset rollout got empty target keys at local rows {missing[:5]}")

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
                if prefix_dones[env_idx] or step_idx >= int(plan.prefix_step_count):
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
                    self.prefix_sft_records[env_idx].append(
                        {
                            "obs_text": prompt_texts[env_idx],
                            "action": expert_action,
                            "think": expert_think,
                            "step_idx": step_idx,
                        }
                    )

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


@dataclass
class MicroGroupResult:
    output: DataProto | None
    repeated_gen_batch: DataProto
    prefix_sft_records: list[list[dict[str, Any]]] | None
    traj_uid: np.ndarray
    episode_rewards: np.ndarray
    gamefiles_by_prompt: list[str]


def _as_list(value, *, cast):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [cast(part) for part in parts]
    return [cast(item) for item in value]


def _make_indices(train_batch_size: int, full_group_size: int, micro_group_size: int) -> list[int]:
    indices: list[int] = []
    for prompt_idx in range(train_batch_size):
        group_start = prompt_idx * full_group_size
        indices.extend(group_start + offset for offset in range(micro_group_size))
    return indices


def _make_gen_batch_with_gamefiles(gen_batch: DataProto, gamefiles: list[str] | None, micro_group_size: int) -> DataProto:
    next_batch = gen_batch.select(deepcopy=True)
    if gamefiles is not None:
        next_batch.non_tensor_batch["env_kwargs"] = np.array(
            [{"gamefile": str(gamefile), "webshop_target_key": str(gamefile)} for gamefile in gamefiles],
            dtype=object,
        )
    return next_batch.repeat(repeat_times=micro_group_size, interleave=True)


def _make_selected_envs(envs, indices: list[int]):
    env_name = str(envs.config.env.env_name).lower()
    if "webshop" in env_name:
        return FixedTargetWebshopEnvironmentManager(envs, indices=indices)
    if "alfworld" in env_name:
        return FixedTargetAlfworldEnvironmentManager(envs, indices=indices)
    raise ValueError(f"TRAPO currently supports ALFWorld and WebShop, got env.env_name={envs.config.env.env_name}")


def _build_sft_batch(collector, result: MicroGroupResult) -> DataProto | None:
    if not collector._prefix_sft_enabled(is_train=True):
        return None
    return collector._build_prefix_sft_batch(
        gen_batch=result.repeated_gen_batch,
        prefix_sft_records=result.prefix_sft_records,
        traj_uid=result.traj_uid,
    )


def _override_group_uid(output: DataProto, prompt_uids: list[str]) -> DataProto:
    prompt_indices = np.asarray(output.non_tensor_batch.get("trapo_prompt_index", []), dtype=np.int64)
    if prompt_indices.size == 0:
        return output
    output.non_tensor_batch["uid"] = np.array([prompt_uids[int(idx)] for idx in prompt_indices], dtype=object)
    return output


def _run_micro_group(
    *,
    collector,
    gen_batch: DataProto,
    actor_rollout_wg,
    envs,
    runtime: TRAPOAlfworldPrefixRuntime,
    micro_group_index: int,
    micro_group_size: int,
    ratio_by_prompt: dict[int, float],
    triggered_by_prompt: dict[int, bool],
    gamefiles: list[str] | None,
    full_group_size: int,
    prompt_uids: list[str],
) -> MicroGroupResult:
    train_batch_size = len(gen_batch)
    repeated_gen_batch = _make_gen_batch_with_gamefiles(gen_batch, gamefiles, micro_group_size)
    runtime.configure_micro_group(
        micro_group_index=micro_group_index,
        active_group_size=micro_group_size,
        ratio_by_group_index=ratio_by_prompt,
        triggered_by_group_index=triggered_by_prompt,
        gamefile_by_group_index={idx: gamefile for idx, gamefile in enumerate(gamefiles or [])},
    )

    selected_envs = _make_selected_envs(
        envs,
        indices=_make_indices(train_batch_size, full_group_size, micro_group_size),
    )
    total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = collector.vanilla_multi_turn_loop(
        gen_batch=repeated_gen_batch,
        actor_rollout_wg=actor_rollout_wg,
        envs=selected_envs,
        is_train=True,
        prefix_runtime_override=runtime,
        collect_rollout_details=False,
    )
    output = None
    if collector._has_policy_rollout_steps(total_batch_list):
        output = collector.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success,
            traj_uid=traj_uid,
            tool_callings=tool_callings,
        )
        output = _override_group_uid(output, prompt_uids)

    gamefiles_by_prompt: list[str] = []
    for prompt_idx in range(train_batch_size):
        plan_idx = prompt_idx * micro_group_size
        if plan_idx < len(runtime.last_plans):
            gamefiles_by_prompt.append(runtime.last_plans[plan_idx].gamefile)
        elif gamefiles is not None:
            gamefiles_by_prompt.append(gamefiles[prompt_idx])
        else:
            gamefiles_by_prompt.append("")

    return MicroGroupResult(
        output=output,
        repeated_gen_batch=repeated_gen_batch,
        prefix_sft_records=getattr(selected_envs, "prefix_sft_records", None),
        traj_uid=np.asarray(traj_uid, dtype=object),
        episode_rewards=np.asarray(episode_rewards, dtype=np.float32),
        gamefiles_by_prompt=gamefiles_by_prompt,
    )


def run_trapo_multi_turn_loop(
    *,
    collector,
    gen_batch: DataProto,
    actor_rollout_wg,
    envs,
    global_step: int | None = None,
    rollout_detail_dump_dir: str | None = None,
    collect_rollout_details: bool = False,
) -> DataProto:
    cfg = collector.config.get("trapo", {})
    if bool(collector.config.algorithm.filter_groups.get("enable", False)):
        raise ValueError("trapo currently requires algorithm.filter_groups.enable=false")

    runtime = getattr(collector, "trapo_runtime", None)
    if runtime is None:
        raise RuntimeError("trapo is enabled but TrajectoryCollector has no trapo_runtime")

    full_group_size = max(int(collector.config.env.rollout.n), 1)
    micro_group_sizes = _as_list(cfg.get("micro_group_sizes", [4, 2, 1, 1]), cast=int)
    prefix_ratios = _as_list(cfg.get("prefix_ratios", [0.0, 0.2, 0.5, 1.0]), cast=float)
    thresholds = _as_list(cfg.get("thresholds", [-1.0, 0.5, 0.7, 0.9]), cast=float)
    if not (len(micro_group_sizes) == len(prefix_ratios) == len(thresholds)):
        raise ValueError("trapo.micro_group_sizes, prefix_ratios, and thresholds must have the same length")
    if sum(micro_group_sizes) != full_group_size:
        raise ValueError(
            f"TRAPO micro-group sizes must sum to env.rollout.n={full_group_size}, got {micro_group_sizes}"
        )

    train_batch_size = len(gen_batch)
    prompt_uids = [f"trapo-{uuid.uuid4()}" for _ in range(train_batch_size)]
    success_threshold = float(cfg.get("success_reward_threshold", 0.0))

    outputs: list[DataProto] = []
    sft_batches: list[DataProto] = []
    successes_by_prompt = [0 for _ in range(train_batch_size)]
    rollouts_by_prompt = [0 for _ in range(train_batch_size)]
    gamefiles: list[str] | None = None
    triggered_counts: list[int] = []

    for micro_idx, (micro_size, prefix_ratio, threshold) in enumerate(zip(micro_group_sizes, prefix_ratios, thresholds)):
        ratio_by_prompt: dict[int, float] = {}
        triggered_by_prompt: dict[int, bool] = {}
        for prompt_idx in range(train_batch_size):
            pass_rate = successes_by_prompt[prompt_idx] / max(1, rollouts_by_prompt[prompt_idx])
            # Match TRAPO Algorithm 1: inject guidance when pass_rate <= threshold.
            triggered = pass_rate <= float(threshold)
            triggered_by_prompt[prompt_idx] = triggered
            ratio_by_prompt[prompt_idx] = float(prefix_ratio) if triggered else 0.0

        result = _run_micro_group(
            collector=collector,
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            runtime=runtime,
            micro_group_index=micro_idx,
            micro_group_size=int(micro_size),
            ratio_by_prompt=ratio_by_prompt,
            triggered_by_prompt=triggered_by_prompt,
            gamefiles=gamefiles,
            full_group_size=full_group_size,
            prompt_uids=prompt_uids,
        )
        if result.output is not None:
            outputs.append(result.output)
        sft_batch = _build_sft_batch(collector, result)
        if sft_batch is not None and len(sft_batch) > 0:
            sft_batches.append(sft_batch)

        if gamefiles is None:
            gamefiles = result.gamefiles_by_prompt

        rewards = result.episode_rewards.reshape(train_batch_size, int(micro_size))
        for prompt_idx in range(train_batch_size):
            successes_by_prompt[prompt_idx] += int(np.sum(rewards[prompt_idx] > success_threshold))
            rollouts_by_prompt[prompt_idx] += int(micro_size)
        triggered_counts.append(sum(1 for value in triggered_by_prompt.values() if value))

    if not outputs:
        raise RuntimeError("trapo produced no rollout outputs")

    output = DataProto.concat(outputs)
    collector.latest_prefix_sft_batch = DataProto.concat(sft_batches) if sft_batches else None
    collector._pending_prefix_sft_batches = []
    collector._last_prefix_sft_records = None
    collector._last_prefix_sft_traj_uid = None
    collector._last_prefix_detail_records = None

    runtime.last_metadata = {
        "global_step": int(global_step) if global_step is not None else None,
        "group_size": int(full_group_size),
        "micro_group_sizes": [int(value) for value in micro_group_sizes],
        "prefix_ratios": [float(value) for value in prefix_ratios],
        "thresholds": [float(value) for value in thresholds],
        "triggered_counts": [int(value) for value in triggered_counts],
        "final_pass_rate_mean": float(np.mean([s / max(1, r) for s, r in zip(successes_by_prompt, rollouts_by_prompt)])),
    }
    return output
