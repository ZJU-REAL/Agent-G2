from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verl import DataProto

from .runtime import BinarySearchAlfworldPrefixRuntime, BinarySearchPrefixPlan


@dataclass
class CandidateResult:
    name: str
    output: DataProto
    repeated_gen_batch: DataProto
    prefix_sft_records: list[list[dict[str, Any]]] | None
    traj_uid: np.ndarray
    episode_rewards: np.ndarray
    plans: list[BinarySearchPrefixPlan]
    success_count_by_group: dict[int, int]
    rollout_count_by_group: dict[int, int]


@dataclass
class GroupSearchState:
    group_index: int
    gamefile: str
    trial_id: str
    total_steps: int
    upper: int
    low: int = 0
    high: int = 0
    selected_step: int | None = None
    selected_status: str | None = None
    selected_candidate: CandidateResult | None = None
    selected_success_count: int | None = None
    visited: dict[int, tuple[int, CandidateResult]] = field(default_factory=dict)


def _target_success_count(group_size: int, cfg) -> int:
    if cfg.get("target_success_count") is not None:
        return int(cfg.get("target_success_count"))
    target_rate = float(cfg.get("target_success_rate", 0.5))
    return int(round(target_rate * group_size))


def _make_gen_batch_with_gamefiles(gen_batch: DataProto, gamefiles: list[str] | None) -> DataProto:
    next_batch = gen_batch.select(deepcopy=True)
    if gamefiles is not None:
        next_batch.non_tensor_batch["env_kwargs"] = np.array(
            [{"gamefile": str(gamefile)} for gamefile in gamefiles],
            dtype=object,
        )
    return next_batch


def _run_candidate(
    *,
    collector,
    gen_batch: DataProto,
    actor_rollout_wg,
    envs,
    runtime: BinarySearchAlfworldPrefixRuntime,
    group_size: int,
    candidate_name: str,
    gamefiles: list[str] | None,
) -> CandidateResult:
    candidate_gen_batch = _make_gen_batch_with_gamefiles(gen_batch, gamefiles)
    repeated_gen_batch = candidate_gen_batch.repeat(repeat_times=group_size, interleave=True)
    candidate_envs = envs
    if gamefiles is not None and "webshop" in str(collector.config.env.env_name).lower():
        from .webshop import FixedTargetWebshopEnvironmentManager

        candidate_envs = FixedTargetWebshopEnvironmentManager(
            envs,
            indices=list(range(len(repeated_gen_batch))),
        )
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

    success_count_by_group: dict[int, int] = {}
    rollout_count_by_group: dict[int, int] = {}
    for group_start in range(0, len(episode_rewards), group_size):
        group_idx = group_start // group_size
        group_rewards = episode_rewards[group_start : group_start + group_size]
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
    )


def _candidate_group_plan(candidate: CandidateResult, group_idx: int, group_size: int) -> BinarySearchPrefixPlan | None:
    plan_idx = group_idx * group_size
    if 0 <= plan_idx < len(candidate.plans):
        return candidate.plans[plan_idx]
    return None


def _select_group_rows(output: DataProto, group_idx: int) -> DataProto | None:
    group_values = np.asarray(output.non_tensor_batch.get("binary_search_group_index", []), dtype=np.int64)
    if group_values.size == 0:
        return None
    indices = np.flatnonzero(group_values == int(group_idx)).astype(np.int64)
    if indices.size == 0:
        return None
    return output[indices]


def _add_selection_metadata(
    data: DataProto,
    *,
    step: int,
    success_count: int,
    target_success_count: int,
    status: str,
    eval_count: int,
    upper: int,
) -> DataProto:
    size = len(data)
    data.non_tensor_batch["binary_search_selected_step_count"] = np.full(size, int(step), dtype=np.int64)
    data.non_tensor_batch["binary_search_selected_success_count"] = np.full(size, int(success_count), dtype=np.int64)
    data.non_tensor_batch["binary_search_target_success_count"] = np.full(size, int(target_success_count), dtype=np.int64)
    data.non_tensor_batch["binary_search_selected_gap"] = np.full(size, abs(int(success_count) - int(target_success_count)), dtype=np.int64)
    data.non_tensor_batch["binary_search_status"] = np.array([status for _ in range(size)], dtype=object)
    data.non_tensor_batch["binary_search_eval_count"] = np.full(size, int(eval_count), dtype=np.int64)
    data.non_tensor_batch["binary_search_upper_step_count"] = np.full(size, int(upper), dtype=np.int64)
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
    start = group_idx * group_size
    stop = min(start + group_size, len(candidate.traj_uid))
    keep_traj_uids = {str(uid) for uid in candidate.traj_uid[start:stop]}
    return collector._build_prefix_sft_batch(
        gen_batch=candidate.repeated_gen_batch,
        prefix_sft_records=candidate.prefix_sft_records,
        traj_uid=candidate.traj_uid,
        keep_traj_uids=keep_traj_uids,
    )


def _choose_closest(state: GroupSearchState, target_success_count: int) -> tuple[int, int, CandidateResult]:
    if not state.visited:
        raise RuntimeError(f"binary search group {state.group_index} has no evaluated candidates")
    step, (success_count, candidate) = min(
        state.visited.items(),
        key=lambda item: (abs(item[1][0] - target_success_count), item[0]),
    )
    return int(step), int(success_count), candidate


def _finalize_state(
    state: GroupSearchState,
    *,
    step: int,
    success_count: int,
    candidate: CandidateResult,
    status: str,
) -> None:
    state.selected_step = int(step)
    state.selected_success_count = int(success_count)
    state.selected_candidate = candidate
    state.selected_status = status


def _record_candidate(
    states: dict[int, GroupSearchState],
    candidate: CandidateResult,
    step_by_group: dict[int, int],
) -> None:
    for group_idx, step in step_by_group.items():
        state = states[group_idx]
        success_count = candidate.success_count_by_group.get(group_idx, 0)
        state.visited[int(step)] = (int(success_count), candidate)


def _run_boundary_first_binary_search_multi_turn_loop(
    *,
    collector,
    gen_batch: DataProto,
    actor_rollout_wg,
    envs,
    global_step: int | None = None,
    rollout_detail_dump_dir: str | None = None,
    collect_rollout_details: bool = False,
) -> DataProto:
    cfg = collector.config.get("binary_search", {})
    if bool(collector.config.algorithm.filter_groups.get("enable", False)):
        raise ValueError("binary_search currently requires algorithm.filter_groups.enable=false")

    runtime = getattr(collector, "binary_search_runtime", None)
    if runtime is None:
        raise RuntimeError("binary_search is enabled but TrajectoryCollector has no binary_search_runtime")

    group_size = max(int(collector.config.env.rollout.n), 1)
    target_success_count = _target_success_count(group_size, cfg)
    max_search_rounds = int(cfg.get("max_search_rounds", 0) or 0)
    train_batch_size = len(gen_batch)

    runtime.configure_steps(default_step_count=0)
    boundary_zero = _run_candidate(
        collector=collector,
        gen_batch=gen_batch,
        actor_rollout_wg=actor_rollout_wg,
        envs=envs,
        runtime=runtime,
        group_size=group_size,
        candidate_name="boundary_zero",
        gamefiles=None,
    )

    gamefiles: list[str] = []
    states: dict[int, GroupSearchState] = {}
    for group_idx in range(train_batch_size):
        plan = _candidate_group_plan(boundary_zero, group_idx, group_size)
        if plan is None:
            raise RuntimeError(f"binary_search failed to read prefix plan for group {group_idx}")
        gamefiles.append(plan.gamefile)
        upper = min(max(int(plan.total_step_count) - 1, 0), max(int(collector.config.env.max_steps) - 1, 0))
        states[group_idx] = GroupSearchState(
            group_index=group_idx,
            gamefile=plan.gamefile,
            trial_id=plan.trial_id,
            total_steps=int(plan.total_step_count),
            upper=upper,
            low=0,
            high=upper,
        )

    _record_candidate(states, boundary_zero, {group_idx: 0 for group_idx in states})

    unresolved: set[int] = set(states.keys())
    for group_idx in list(unresolved):
        state = states[group_idx]
        success_count = boundary_zero.success_count_by_group.get(group_idx, 0)
        if state.upper <= 0:
            _finalize_state(state, step=0, success_count=success_count, candidate=boundary_zero, status="short_fallback")
            unresolved.remove(group_idx)
        elif success_count == target_success_count:
            _finalize_state(state, step=0, success_count=success_count, candidate=boundary_zero, status="hit")
            unresolved.remove(group_idx)
        elif success_count > target_success_count:
            _finalize_state(state, step=0, success_count=success_count, candidate=boundary_zero, status="easy_censored")
            unresolved.remove(group_idx)

    if unresolved:
        step_by_group = {group_idx: states[group_idx].upper for group_idx in unresolved}
        runtime.configure_steps(
            default_step_count=0,
            step_by_group_index=step_by_group,
        )
        boundary_upper = _run_candidate(
            collector=collector,
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            runtime=runtime,
            group_size=group_size,
            candidate_name="boundary_upper",
            gamefiles=gamefiles,
        )
        _record_candidate(states, boundary_upper, step_by_group)

        for group_idx in list(unresolved):
            state = states[group_idx]
            success_count = boundary_upper.success_count_by_group.get(group_idx, 0)
            if success_count == target_success_count:
                _finalize_state(state, step=state.upper, success_count=success_count, candidate=boundary_upper, status="hit")
                unresolved.remove(group_idx)
            elif success_count < target_success_count:
                _finalize_state(state, step=state.upper, success_count=success_count, candidate=boundary_upper, status="hard_censored")
                unresolved.remove(group_idx)
            else:
                state.low = 1
                state.high = state.upper - 1
                if state.low > state.high:
                    step, closest_success, closest_candidate = _choose_closest(state, target_success_count)
                    _finalize_state(state, step=step, success_count=closest_success, candidate=closest_candidate, status="closest")
                    unresolved.remove(group_idx)

    round_idx = 0
    while unresolved:
        round_idx += 1
        if max_search_rounds > 0 and round_idx > max_search_rounds:
            for group_idx in list(unresolved):
                state = states[group_idx]
                step, success_count, candidate = _choose_closest(state, target_success_count)
                _finalize_state(state, step=step, success_count=success_count, candidate=candidate, status="closest_max_rounds")
                unresolved.remove(group_idx)
            break

        step_by_group: dict[int, int] = {}
        for group_idx in list(unresolved):
            state = states[group_idx]
            if state.low > state.high:
                step, success_count, candidate = _choose_closest(state, target_success_count)
                _finalize_state(state, step=step, success_count=success_count, candidate=candidate, status="closest")
                unresolved.remove(group_idx)
                continue
            mid = int(math.floor((state.low + state.high) / 2))
            if mid in state.visited:
                step, success_count, candidate = _choose_closest(state, target_success_count)
                _finalize_state(state, step=step, success_count=success_count, candidate=candidate, status="closest_repeat")
                unresolved.remove(group_idx)
                continue
            step_by_group[group_idx] = mid

        if not step_by_group:
            continue

        runtime.configure_steps(
            default_step_count=0,
            step_by_group_index=step_by_group,
        )
        candidate = _run_candidate(
            collector=collector,
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            runtime=runtime,
            group_size=group_size,
            candidate_name=f"binary_round_{round_idx}",
            gamefiles=gamefiles,
        )
        _record_candidate(states, candidate, step_by_group)

        for group_idx, mid in step_by_group.items():
            if group_idx not in unresolved:
                continue
            state = states[group_idx]
            success_count = candidate.success_count_by_group.get(group_idx, 0)
            if success_count == target_success_count:
                _finalize_state(state, step=mid, success_count=success_count, candidate=candidate, status="hit")
                unresolved.remove(group_idx)
            elif success_count < target_success_count:
                state.low = mid + 1
            else:
                state.high = mid - 1

    selected_outputs: list[DataProto] = []
    selected_sft_batches: list[DataProto] = []
    for group_idx in sorted(states):
        state = states[group_idx]
        if state.selected_candidate is None or state.selected_step is None or state.selected_success_count is None:
            step, success_count, candidate = _choose_closest(state, target_success_count)
            _finalize_state(state, step=step, success_count=success_count, candidate=candidate, status="closest_unfinished")

        selected = _select_group_rows(state.selected_candidate.output, group_idx)
        if selected is None or len(selected) == 0:
            continue
        selected = _add_selection_metadata(
            selected,
            step=int(state.selected_step),
            success_count=int(state.selected_success_count),
            target_success_count=target_success_count,
            status=str(state.selected_status),
            eval_count=len(state.visited),
            upper=state.upper,
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
        raise RuntimeError("binary_search produced no selected rollout rows for training")

    output = DataProto.concat(selected_outputs)
    collector.latest_prefix_sft_batch = DataProto.concat(selected_sft_batches) if selected_sft_batches else None
    collector._pending_prefix_sft_batches = []
    collector._last_prefix_sft_records = None
    collector._last_prefix_sft_traj_uid = None
    collector._last_prefix_detail_records = None

    statuses = [str(states[idx].selected_status) for idx in sorted(states)]
    selected_steps = [int(states[idx].selected_step or 0) for idx in sorted(states)]
    selected_successes = [int(states[idx].selected_success_count or 0) for idx in sorted(states)]
    runtime.last_metadata = {
        "global_step": int(global_step) if global_step is not None else None,
        "search_mode": "boundary_first",
        "target_success_count": int(target_success_count),
        "group_size": int(group_size),
        "num_groups": int(len(states)),
        "mean_selected_step": float(np.mean(selected_steps)) if selected_steps else 0.0,
        "mean_selected_success_count": float(np.mean(selected_successes)) if selected_successes else 0.0,
        "statuses": {status: statuses.count(status) for status in sorted(set(statuses))},
    }
    return output


def _finalize_selected_outputs(
    *,
    collector,
    states: dict[int, GroupSearchState],
    target_success_count: int,
    group_size: int,
) -> DataProto:
    selected_outputs: list[DataProto] = []
    selected_sft_batches: list[DataProto] = []
    for group_idx in sorted(states):
        state = states[group_idx]
        if state.selected_candidate is None or state.selected_step is None or state.selected_success_count is None:
            step, success_count, candidate = _choose_closest(state, target_success_count)
            _finalize_state(state, step=step, success_count=success_count, candidate=candidate, status="closest_unfinished")

        selected = _select_group_rows(state.selected_candidate.output, group_idx)
        if selected is None or len(selected) == 0:
            continue
        selected = _add_selection_metadata(
            selected,
            step=int(state.selected_step),
            success_count=int(state.selected_success_count),
            target_success_count=target_success_count,
            status=str(state.selected_status),
            eval_count=len(state.visited),
            upper=state.upper,
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
        raise RuntimeError("binary_search produced no selected rollout rows for training")

    output = DataProto.concat(selected_outputs)
    collector.latest_prefix_sft_batch = DataProto.concat(selected_sft_batches) if selected_sft_batches else None
    collector._pending_prefix_sft_batches = []
    collector._last_prefix_sft_records = None
    collector._last_prefix_sft_traj_uid = None
    collector._last_prefix_detail_records = None
    return output


def _run_pure_binary_search_multi_turn_loop(
    *,
    collector,
    gen_batch: DataProto,
    actor_rollout_wg,
    envs,
    global_step: int | None = None,
    rollout_detail_dump_dir: str | None = None,
    collect_rollout_details: bool = False,
) -> DataProto:
    cfg = collector.config.get("binary_search", {})
    if bool(collector.config.algorithm.filter_groups.get("enable", False)):
        raise ValueError("binary_search currently requires algorithm.filter_groups.enable=false")

    runtime = getattr(collector, "binary_search_runtime", None)
    if runtime is None:
        raise RuntimeError("binary_search is enabled but TrajectoryCollector has no binary_search_runtime")

    group_size = max(int(collector.config.env.rollout.n), 1)
    target_success_count = _target_success_count(group_size, cfg)
    max_search_rounds = int(cfg.get("max_search_rounds", 0) or 0)
    train_batch_size = len(gen_batch)

    runtime.configure_midpoint_steps()
    first_candidate = _run_candidate(
        collector=collector,
        gen_batch=gen_batch,
        actor_rollout_wg=actor_rollout_wg,
        envs=envs,
        runtime=runtime,
        group_size=group_size,
        candidate_name="binary_round_1",
        gamefiles=None,
    )

    gamefiles: list[str] = []
    states: dict[int, GroupSearchState] = {}
    first_step_by_group: dict[int, int] = {}
    for group_idx in range(train_batch_size):
        plan = _candidate_group_plan(first_candidate, group_idx, group_size)
        if plan is None:
            raise RuntimeError(f"binary_search failed to read prefix plan for group {group_idx}")
        gamefiles.append(plan.gamefile)
        upper = min(max(int(plan.total_step_count) - 1, 0), max(int(collector.config.env.max_steps) - 1, 0))
        first_step = min(max(int(plan.prefix_step_count), 0), upper)
        states[group_idx] = GroupSearchState(
            group_index=group_idx,
            gamefile=plan.gamefile,
            trial_id=plan.trial_id,
            total_steps=int(plan.total_step_count),
            upper=upper,
            low=0,
            high=upper,
        )
        first_step_by_group[group_idx] = first_step

    _record_candidate(states, first_candidate, first_step_by_group)

    unresolved: set[int] = set(states.keys())
    for group_idx in list(unresolved):
        state = states[group_idx]
        mid = first_step_by_group[group_idx]
        success_count = first_candidate.success_count_by_group.get(group_idx, 0)
        if success_count == target_success_count:
            _finalize_state(state, step=mid, success_count=success_count, candidate=first_candidate, status="hit")
            unresolved.remove(group_idx)
        elif success_count < target_success_count:
            state.low = mid + 1
        else:
            state.high = mid - 1

    round_idx = 1
    while unresolved:
        if max_search_rounds > 0 and round_idx >= max_search_rounds:
            for group_idx in list(unresolved):
                state = states[group_idx]
                step, success_count, candidate = _choose_closest(state, target_success_count)
                _finalize_state(state, step=step, success_count=success_count, candidate=candidate, status="closest_max_rounds")
                unresolved.remove(group_idx)
            break

        step_by_group: dict[int, int] = {}
        for group_idx in list(unresolved):
            state = states[group_idx]
            if state.low > state.high:
                step, success_count, candidate = _choose_closest(state, target_success_count)
                _finalize_state(state, step=step, success_count=success_count, candidate=candidate, status="closest")
                unresolved.remove(group_idx)
                continue
            mid = int(math.floor((state.low + state.high) / 2))
            if mid in state.visited:
                step, success_count, candidate = _choose_closest(state, target_success_count)
                _finalize_state(state, step=step, success_count=success_count, candidate=candidate, status="closest_repeat")
                unresolved.remove(group_idx)
                continue
            step_by_group[group_idx] = mid

        if not step_by_group:
            continue

        round_idx += 1
        runtime.configure_steps(
            default_step_count=0,
            step_by_group_index=step_by_group,
        )
        candidate = _run_candidate(
            collector=collector,
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            runtime=runtime,
            group_size=group_size,
            candidate_name=f"binary_round_{round_idx}",
            gamefiles=gamefiles,
        )
        _record_candidate(states, candidate, step_by_group)

        for group_idx, mid in step_by_group.items():
            if group_idx not in unresolved:
                continue
            state = states[group_idx]
            success_count = candidate.success_count_by_group.get(group_idx, 0)
            if success_count == target_success_count:
                _finalize_state(state, step=mid, success_count=success_count, candidate=candidate, status="hit")
                unresolved.remove(group_idx)
            elif success_count < target_success_count:
                state.low = mid + 1
            else:
                state.high = mid - 1

    output = _finalize_selected_outputs(
        collector=collector,
        states=states,
        target_success_count=target_success_count,
        group_size=group_size,
    )

    statuses = [str(states[idx].selected_status) for idx in sorted(states)]
    selected_steps = [int(states[idx].selected_step or 0) for idx in sorted(states)]
    selected_successes = [int(states[idx].selected_success_count or 0) for idx in sorted(states)]
    eval_counts = [int(len(states[idx].visited)) for idx in sorted(states)]
    runtime.last_metadata = {
        "global_step": int(global_step) if global_step is not None else None,
        "search_mode": "pure_binary",
        "target_success_count": int(target_success_count),
        "group_size": int(group_size),
        "num_groups": int(len(states)),
        "mean_selected_step": float(np.mean(selected_steps)) if selected_steps else 0.0,
        "mean_selected_success_count": float(np.mean(selected_successes)) if selected_successes else 0.0,
        "mean_eval_count": float(np.mean(eval_counts)) if eval_counts else 0.0,
        "statuses": {status: statuses.count(status) for status in sorted(set(statuses))},
    }
    return output


def run_binary_search_multi_turn_loop(
    *,
    collector,
    gen_batch: DataProto,
    actor_rollout_wg,
    envs,
    global_step: int | None = None,
    rollout_detail_dump_dir: str | None = None,
    collect_rollout_details: bool = False,
) -> DataProto:
    cfg = collector.config.get("binary_search", {})
    search_mode = str(cfg.get("search_mode", cfg.get("mode", "pure_binary"))).lower()
    if search_mode in {"pure_binary", "midpoint_first", "binary"}:
        return _run_pure_binary_search_multi_turn_loop(
            collector=collector,
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            global_step=global_step,
            rollout_detail_dump_dir=rollout_detail_dump_dir,
            collect_rollout_details=collect_rollout_details,
        )
    if search_mode in {"boundary_first", "boundary", "legacy"}:
        return _run_boundary_first_binary_search_multi_turn_loop(
            collector=collector,
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            global_step=global_step,
            rollout_detail_dump_dir=rollout_detail_dump_dir,
            collect_rollout_details=collect_rollout_details,
        )
    raise ValueError(f"Unknown binary_search.search_mode: {search_mode}")
