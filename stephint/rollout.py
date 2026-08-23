from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from agent_system.environments.base import to_numpy
from agent_system.environments.env_manager import set_gamefile
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.model import compute_position_id_with_mask


def _forced_response_text(action: str, think: str = "") -> str:
    action = str(action).strip()
    think = str(think).strip()
    if think:
        return f"<think>{think}</think>\n<action>{action}</action>"
    return f"<think></think>\n<action>{action}</action>"


def _is_forced_action_valid(action: str, available_actions: list[str]) -> bool:
    normalized_action = str(action).strip().lower()
    normalized_available = {str(item).strip().lower() for item in available_actions}
    if normalized_action in normalized_available:
        return True
    return normalized_action.startswith("search[") and any(
        item.startswith("search[") for item in normalized_available
    )


def _is_webshop_envs(envs) -> bool:
    return hasattr(envs, "last_infos") and hasattr(envs, "format_avail_actions")


def _build_prefix_prompt_texts(envs, step_idx: int) -> list[str]:
    if _is_webshop_envs(envs):
        init_prompt = all(len(envs.memory[env_idx]) == 0 for env_idx in range(len(envs.pre_text_obs)))
        return envs.build_text_obs(envs.pre_text_obs, envs.last_infos, init=init_prompt)
    return envs.build_text_obs(
        envs.pre_text_obs,
        envs.envs.get_admissible_commands,
        init=(step_idx == 0),
    )


def _available_actions_snapshot(envs) -> list[list[str]]:
    if _is_webshop_envs(envs):
        return [
            envs.format_avail_actions(info.get("available_actions", {}))
            if isinstance(info, dict)
            else []
            for info in envs.last_infos
        ]
    return [
        list(commands) if commands is not None else []
        for commands in envs.envs.get_admissible_commands
    ]


def _build_prefix_next_observations(envs) -> dict[str, Any]:
    if _is_webshop_envs(envs):
        init_prompt = all(len(envs.memory[env_idx]) == 0 for env_idx in range(len(envs.pre_text_obs)))
        return {
            "text": envs.build_text_obs(envs.pre_text_obs, envs.last_infos, init=init_prompt),
            "image": None,
            "anchor": envs.pre_text_obs.copy(),
        }

    full_text_obs = envs.build_text_obs(envs.pre_text_obs, envs.envs.get_admissible_commands)
    image_obs = envs.envs.getobs() if getattr(envs.envs, "multi_modal", False) else None
    return {"text": full_text_obs, "image": image_obs, "anchor": envs.pre_text_obs.copy()}


def _tokenize_forced_step(
    *,
    collector,
    obs_text: str,
    response_text: str,
    data_source: Any,
    meta: dict[str, Any],
    protect_negative_advantage: bool,
) -> dict[str, Any]:
    tokenizer = collector.tokenizer
    apply_chat_template_kwargs = collector.config.data.get("apply_chat_template_kwargs", {})
    max_prompt_length = int(collector.config.data.max_prompt_length)
    max_response_length = int(collector.config.data.max_response_length)

    chat = np.array([{"content": str(obs_text), "role": "user"}])
    prompt = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
        **apply_chat_template_kwargs,
    )
    prompt_ids, prompt_attention_mask = verl_F.tokenize_and_postprocess_data(
        prompt=prompt,
        tokenizer=tokenizer,
        max_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation=collector.config.data.truncation,
    )
    prompt_position_ids = compute_position_id_with_mask(prompt_attention_mask)

    response_ids_list = tokenizer.encode(response_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        response_ids_list.append(tokenizer.eos_token_id)
    response_ids = torch.tensor(response_ids_list, dtype=torch.long).unsqueeze(0)
    response_attention_mask = torch.ones_like(response_ids)
    response_ids, response_attention_mask = verl_F.postprocess_data(
        input_ids=response_ids,
        attention_mask=response_attention_mask,
        max_length=max_response_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=False,
        truncation="right",
    )

    response_position_delta = torch.arange(1, max_response_length + 1).unsqueeze(0)
    response_position_ids = prompt_position_ids[:, -1:] + response_position_delta

    input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
    position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

    protected_mask = response_attention_mask[0].bool()
    if not protect_negative_advantage:
        protected_mask = torch.zeros_like(protected_mask, dtype=torch.bool)

    row: dict[str, Any] = {
        "prompts": prompt_ids[0],
        "responses": response_ids[0],
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "position_ids": position_ids[0],
        "step_hint_protected_mask": protected_mask,
        "data_source": data_source,
    }
    row.update(meta)
    return row


def _build_forced_row(
    *,
    collector,
    gen_batch: DataProto,
    env_idx: int,
    traj_uid: str,
    uid: str,
    obs_text: str,
    action: str,
    think: str,
    reward: float,
    done: bool,
    info: dict[str, Any],
    plan,
    step_idx: int,
    available_actions: list[str],
    next_observation: str | None,
) -> dict[str, Any]:
    response_text = _forced_response_text(action=action, think=think)
    data_source = gen_batch.non_tensor_batch["data_source"][env_idx]
    is_action_valid = _is_forced_action_valid(action=action, available_actions=available_actions)

    row = _tokenize_forced_step(
        collector=collector,
        obs_text=obs_text,
        response_text=response_text,
        data_source=data_source,
        protect_negative_advantage=bool(collector.config.get("step_hint", {}).get("protect_negative_advantage", True)),
        meta={
            "uid": uid,
            "traj_uid": traj_uid,
            "rewards": np.array(reward, dtype=np.float32),
            "active_masks": np.array(True, dtype=bool),
            "is_action_valid": np.array(is_action_valid, dtype=bool),
            "step_hint_forced": True,
            "step_hint_reference": bool(plan.is_reference),
            "step_hint_is_reference": bool(plan.is_reference),
            "step_hint_level": int(plan.level),
            "step_hint_slot_type": str(plan.slot_type),
            "step_hint_trial_id": str(plan.trial_id),
            "step_hint_gamefile": str(plan.gamefile),
            "step_hint_matched": bool(plan.matched),
            "step_hint_task_type": str(plan.task_type),
            "step_hint_difficulty_group": int(plan.difficulty_group),
            "step_hint_prefix_step_count": int(plan.prefix_step_count),
            "step_hint_total_step_count": int(plan.total_step_count),
            "step_hint_prefix_token_count": int(plan.prefix_token_count),
            "step_hint_total_token_count": int(plan.total_token_count),
            "anchor_obs": obs_text,
            "index": int(env_idx),
        },
    )
    row["_rollout_detail_step"] = int(step_idx)
    row["_rollout_detail_task"] = getattr(plan, "task_type", "")
    row["_rollout_detail_observation"] = obs_text
    row["_rollout_detail_prompt_observation"] = obs_text
    row["_rollout_detail_available_actions"] = available_actions
    row["_rollout_detail_info"] = info
    row["_rollout_detail_done"] = bool(done)
    row["_rollout_detail_reward"] = float(reward)
    row["_rollout_detail_next_observation"] = next_observation
    return row


def _add_policy_step_hint_fields(batch_list: list[dict[str, Any]], plans, batch_size: int) -> None:
    for env_idx in range(batch_size):
        plan = plans[env_idx]
        response = batch_list[env_idx].get("responses")
        if isinstance(response, torch.Tensor):
            mask = torch.zeros_like(response, dtype=torch.bool)
        else:
            mask = torch.zeros(0, dtype=torch.bool)
        batch_list[env_idx].pop("rollout_log_probs", None)
        batch_list[env_idx]["step_hint_protected_mask"] = mask
        batch_list[env_idx]["step_hint_forced"] = False
        batch_list[env_idx]["step_hint_reference"] = False
        batch_list[env_idx]["step_hint_is_reference"] = False
        batch_list[env_idx]["step_hint_level"] = int(plan.level)
        batch_list[env_idx]["step_hint_slot_type"] = str(plan.slot_type)
        batch_list[env_idx]["step_hint_trial_id"] = str(plan.trial_id)
        batch_list[env_idx]["step_hint_gamefile"] = str(plan.gamefile)
        batch_list[env_idx]["step_hint_matched"] = bool(plan.matched)
        batch_list[env_idx]["step_hint_task_type"] = str(plan.task_type)
        batch_list[env_idx]["step_hint_difficulty_group"] = int(plan.difficulty_group)
        batch_list[env_idx]["step_hint_prefix_step_count"] = int(plan.prefix_step_count)
        batch_list[env_idx]["step_hint_total_step_count"] = int(plan.total_step_count)
        batch_list[env_idx]["step_hint_prefix_token_count"] = int(plan.prefix_token_count)
        batch_list[env_idx]["step_hint_total_token_count"] = int(plan.total_token_count)


def _drop_generation_only_fields(total_batch_list: list[list[dict[str, Any]]]) -> None:
    for trajectory_steps in total_batch_list:
        for row in trajectory_steps:
            row.pop("raw_prompt", None)
            row.pop("raw_prompt_ids", None)


def _is_active_row(row: dict[str, Any]) -> bool:
    value = row.get("active_masks", True)
    if isinstance(value, np.ndarray):
        return bool(value.item()) if value.shape == () else bool(value.reshape(-1)[0])
    return bool(value)


def _validate_active_step_keys(total_batch_list: list[list[dict[str, Any]]]) -> None:
    rows = [
        {key: value for key, value in row.items() if not str(key).startswith("_rollout_detail_")}
        for trajectory_steps in total_batch_list
        for row in trajectory_steps
        if _is_active_row(row)
    ]
    if not rows:
        return

    required_keys = set(rows[0])
    for row in rows[1:]:
        required_keys &= set(row)
    all_keys = set().union(*(set(row) for row in rows))
    partial_keys = sorted(all_keys - required_keys)
    if not partial_keys:
        return

    examples: dict[str, list[int]] = {}
    for key in partial_keys:
        missing = [idx for idx, row in enumerate(rows) if key not in row]
        examples[key] = missing[:5]
    raise RuntimeError(
        "StepHint active rollout rows have inconsistent fields before collation: "
        f"{examples}. Drop generation-only fields or add the field to forced and policy rows."
    )


def _replay_forced_prefix(
    *,
    collector,
    gen_batch: DataProto,
    envs,
    plans,
    traj_uid: np.ndarray,
    uid_batch: np.ndarray,
    total_batch_list: list[list[dict[str, Any]]],
    total_infos: list[list[dict[str, Any]]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    batch_size = len(plans)
    prefix_rewards = np.zeros(batch_size, dtype=np.float32)
    prefix_lengths = np.zeros(batch_size, dtype=np.float32)
    prefix_dones = np.zeros(batch_size, dtype=bool)
    envs.prefix_terminal_infos = [None for _ in range(batch_size)]
    envs.prefix_sft_records = None
    envs.prefix_detail_records = None

    max_prefix_steps = max((int(plan.prefix_step_count) for plan in plans), default=0)
    for step_idx in range(max_prefix_steps):
        selected_indices: list[int] = []
        selected_actions: list[str] = []
        selected_thinks: list[str] = []
        for env_idx, plan in enumerate(plans):
            if prefix_dones[env_idx] or step_idx >= int(plan.prefix_step_count):
                continue
            selected_indices.append(env_idx)
            selected_actions.append(str(plan.prefix_actions[step_idx]))
            selected_thinks.append(str(plan.prefix_thinks[step_idx]) if step_idx < len(plan.prefix_thinks) else "")

        if not selected_indices:
            continue

        prompt_texts = _build_prefix_prompt_texts(envs, step_idx=step_idx)
        available_actions_snapshot = _available_actions_snapshot(envs)
        previous_text_obs = [envs.pre_text_obs[env_idx] for env_idx in selected_indices]
        step_result = envs.envs.step_selected(selected_indices, selected_actions)
        if len(step_result) == 5:
            text_obs, _, rewards, dones, infos = step_result
        else:
            text_obs, rewards, dones, infos = step_result
            text_obs = envs.format_obs(text_obs) if hasattr(envs, "format_obs") else text_obs
        if infos and infos[0].get("extra.gamefile") is None and hasattr(envs, "gamefile"):
            infos = set_gamefile(infos, [envs.gamefile[idx] for idx in selected_indices])
        envs.memory.store_selected({"text_obs": previous_text_obs, "action": selected_actions}, selected_indices)

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        for local_idx, env_idx in enumerate(selected_indices):
            reward_value = float(rewards[local_idx])
            info = infos[local_idx]
            envs.pre_text_obs[env_idx] = text_obs[local_idx]
            if hasattr(envs, "last_infos"):
                envs.last_infos[env_idx] = info
            prefix_rewards[env_idx] += reward_value
            prefix_lengths[env_idx] += 1.0
            prefix_dones[env_idx] = prefix_dones[env_idx] or bool(dones[local_idx])
            envs.prefix_terminal_infos[env_idx] = info

            row = _build_forced_row(
                collector=collector,
                gen_batch=gen_batch,
                env_idx=env_idx,
                traj_uid=str(traj_uid[env_idx]),
                uid=str(uid_batch[env_idx]),
                obs_text=prompt_texts[env_idx],
                action=selected_actions[local_idx],
                think=selected_thinks[local_idx],
                reward=reward_value,
                done=bool(dones[local_idx]),
                info=info,
                plan=plans[env_idx],
                step_idx=step_idx,
                available_actions=available_actions_snapshot[env_idx],
                next_observation=text_obs[local_idx],
            )
            total_batch_list[env_idx].append(row)
            total_infos[env_idx].append(info)

    for env_idx, plan in enumerate(plans):
        if plan.is_reference:
            prefix_dones[env_idx] = True

    return _build_prefix_next_observations(envs), prefix_rewards, prefix_lengths, prefix_dones


def stephint_multi_turn_loop(
    collector,
    gen_batch: DataProto,
    actor_rollout_wg,
    envs,
    is_train: bool = True,
    prefix_runtime_override=None,
    global_step: int | None = None,
    rollout_detail_dump_dir: str | None = None,
    collect_rollout_details: bool = False,
) -> DataProto:
    if not is_train:
        return collector._stephint_original_multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            is_train=is_train,
            prefix_runtime_override=prefix_runtime_override,
            global_step=global_step,
            rollout_detail_dump_dir=rollout_detail_dump_dir,
            collect_rollout_details=collect_rollout_details,
        )
    if bool(collector.config.algorithm.filter_groups.get("enable", False)):
        raise ValueError("Strict StepHint currently requires algorithm.filter_groups.enable=false")
    if bool(collector.config.actor_rollout_ref.rollout.multi_turn.get("enable", False)):
        raise ValueError("Strict StepHint ALFWorld replay expects actor_rollout_ref.rollout.multi_turn.enable=false")

    runtime = getattr(collector, "stephint_runtime", None)
    if runtime is None:
        raise RuntimeError("step_hint is enabled but TrajectoryCollector has no stephint_runtime")

    collector.latest_prefix_sft_batch = None
    collector._pending_prefix_sft_batches = []
    should_dump_full_rollout_details = collector._should_dump_full_rollout_detail(
        dump_dir=rollout_detail_dump_dir,
        global_step=global_step,
    )
    should_dump_sft_pairs = False
    should_collect_rollout_details = bool(collect_rollout_details) or should_dump_full_rollout_details or should_dump_sft_pairs

    group_size = max(int(collector.config.env.rollout.n), 1)
    gen_batch = gen_batch.repeat(repeat_times=group_size, interleave=True)
    batch_size = len(gen_batch.batch)
    obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop("env_kwargs", None))
    obs_len = len(obs["text"]) if obs.get("text", None) is not None else len(obs["image"])
    assert batch_size == obs_len, f"gen_batch size {batch_size} does not match obs size {obs_len}"

    plans = runtime.build_prefix_plans(
        infos=infos,
        is_train=True,
        group_size=group_size,
        share_within_group=True,
    )
    prefix_meta = runtime.build_batch_metadata(plans)
    expected = runtime.expected_group_size()
    runtime.last_metadata = {
        "global_step": int(global_step) if global_step is not None else None,
        "group_size": int(group_size),
        "expected_group_size": int(expected),
        "num_plans": int(len(plans)),
    }

    uid_batch = []
    for i in range(batch_size):
        if i % group_size == 0:
            uid = str(uuid.uuid4())
        uid_batch.append(uid)
    uid_batch = np.array(uid_batch, dtype=object)
    traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)

    total_batch_list: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
    total_infos: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
    obs, prefix_rewards, prefix_lengths, prefix_dones = _replay_forced_prefix(
        collector=collector,
        gen_batch=gen_batch,
        envs=envs,
        plans=plans,
        traj_uid=traj_uid,
        uid_batch=uid_batch,
        total_batch_list=total_batch_list,
        total_infos=total_infos,
    )

    remaining_step_budget = np.maximum(
        int(collector.config.env.max_steps) - prefix_lengths.astype(np.int32),
        0,
    ).astype(np.int32)
    is_done = np.logical_or(prefix_dones.copy(), remaining_step_budget <= 0)
    episode_lengths = prefix_lengths.copy()
    episode_rewards = prefix_rewards.copy()
    tool_callings = np.zeros(batch_size, dtype=np.float32)

    for step_idx in range(int(collector.config.env.max_steps)):
        active_masks = np.logical_not(is_done)
        if not active_masks.any():
            break

        batch = collector.preprocess_batch(gen_batch=gen_batch, obs=obs)
        non_tensor_pop_keys = ["raw_prompt_ids"]
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_pop_keys.append("raw_prompt")
        batch_input = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=non_tensor_pop_keys,
        )
        batch_input.meta_info = gen_batch.meta_info
        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
        batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

        batch.non_tensor_batch["uid"] = uid_batch
        batch.non_tensor_batch["traj_uid"] = traj_uid
        batch.non_tensor_batch.update(prefix_meta)
        batch = batch.union(batch_output)

        text_actions = collector.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        current_infos = getattr(envs, "last_infos", None)
        current_tasks = getattr(envs, "tasks", None)
        next_obs, rewards, dones, infos = envs.step(text_actions)
        if len(rewards.shape) == 2:
            rewards = rewards.squeeze(1)
        if len(dones.shape) == 2:
            dones = dones.squeeze(1)

        if "is_action_valid" in infos[0]:
            batch.non_tensor_batch["is_action_valid"] = np.array([info["is_action_valid"] for info in infos], dtype=bool)
        else:
            batch.non_tensor_batch["is_action_valid"] = np.ones(batch_size, dtype=bool)
        if "tool_calling" in infos[0]:
            tool_callings[active_masks] += np.array([info["tool_calling"] for info in infos], dtype=np.float32)[active_masks]
        episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
        episode_lengths[active_masks] += 1
        batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
        batch.non_tensor_batch["active_masks"] = torch_to_numpy(active_masks, is_object=True)

        batch_list: list[dict[str, Any]] = to_list_of_dict(batch)
        _add_policy_step_hint_fields(batch_list=batch_list, plans=plans, batch_size=batch_size)

        for i in range(batch_size):
            current_info = current_infos[i] if isinstance(current_infos, list) and i < len(current_infos) else {}
            if not isinstance(current_info, dict):
                current_info = {}
            current_available_actions = current_info.get("available_actions", current_info.get("admissible_commands"))
            current_task = current_tasks[i] if isinstance(current_tasks, list) and i < len(current_tasks) else None
            current_observation = obs["anchor"][i] if obs.get("anchor", None) is not None else obs["text"][i]
            current_prompt_observation = obs["text"][i] if obs.get("text", None) is not None else None
            batch_list[i]["_rollout_detail_step"] = int(prefix_lengths[i] + step_idx)
            batch_list[i]["_rollout_detail_task"] = current_task
            batch_list[i]["_rollout_detail_observation"] = current_observation
            batch_list[i]["_rollout_detail_prompt_observation"] = current_prompt_observation
            batch_list[i]["_rollout_detail_available_actions"] = current_available_actions
            batch_list[i]["_rollout_detail_info"] = infos[i]
            batch_list[i]["_rollout_detail_done"] = bool(dones[i])
            batch_list[i]["_rollout_detail_reward"] = rewards[i]
            if next_obs.get("anchor", None) is not None:
                batch_list[i]["_rollout_detail_next_observation"] = next_obs["anchor"][i]
            total_batch_list[i].append(batch_list[i])
            total_infos[i].append(infos[i])

        remaining_step_budget[active_masks] -= 1
        is_done = np.logical_or(is_done, dones)
        is_done = np.logical_or(is_done, remaining_step_budget <= 0)
        obs = next_obs
        if is_done.all():
            break

    success = envs.success_evaluator(
        total_infos=total_infos,
        total_batch_list=total_batch_list,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
    )

    rollout_details_payload = None
    if should_collect_rollout_details:
        rollout_details_payload = collector._build_rollout_details_payload(
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success,
            traj_uid=traj_uid,
            tool_callings=tool_callings,
            global_step=global_step,
            is_train=is_train,
            prefix_detail_records=None,
        )
    collector._dump_rollout_details_payload(
        payload=rollout_details_payload,
        dump_dir=rollout_detail_dump_dir,
        global_step=global_step,
    )

    _drop_generation_only_fields(total_batch_list)
    _validate_active_step_keys(total_batch_list)
    output = collector.gather_rollout_data(
        total_batch_list=total_batch_list,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        success=success,
        traj_uid=traj_uid,
        tool_callings=tool_callings,
    )
    if collect_rollout_details and rollout_details_payload is not None:
        output.meta_info["rollout_details"] = rollout_details_payload
    return output
