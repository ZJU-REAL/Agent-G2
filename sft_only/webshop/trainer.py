from __future__ import annotations

from typing import Any, Dict

from gmsv.alfworld import PrefixPlan
from gmsv.webshop import normalize_target_key, target_key_to_string
from sft_only.trainer import SFTOnlyTrainer


class WebshopSFTOnlyTrainer(SFTOnlyTrainer):
    """Replay full WebShop expert trajectories and train only with SFT loss."""

    def _full_expert_plan(self, info: Dict[str, Any]) -> PrefixPlan:
        prefix_runtime = getattr(self.traj_collector, "gmsv_runtime", None)
        if prefix_runtime is None:
            raise RuntimeError("WebShop SFT-only trainer requires a GMSV runtime.")
        prefix_runtime.ensure_initialized()

        target_key = normalize_target_key(info.get("webshop_target_key"))
        trial_id = target_key_to_string(target_key)
        trajectory = prefix_runtime.store.get(target_key)
        if trajectory is None:
            if bool(prefix_runtime.gmsv_cfg.get("strict_expert_match", False)):
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

        keep_steps = min(int(trajectory.action_count), max(int(self.config.env.max_steps), 0))
        prefix_actions = trajectory.actions[:keep_steps]
        prefix_thinks = trajectory.thinks[:keep_steps] if trajectory.thinks else []
        return PrefixPlan(
            trial_id=trajectory.trial_id,
            matched=True,
            task_type=trajectory.task_type,
            difficulty_group=trajectory.difficulty_group,
            mu=1.0,
            sigma=0.0,
            sampled_ratio=1.0,
            clipped_ratio=1.0,
            prefix_actions=prefix_actions,
            prefix_thinks=prefix_thinks,
            prefix_step_count=keep_steps,
            prefix_token_count=len(trajectory.full_prefix_token_ids),
            total_step_count=trajectory.action_count,
            total_token_count=len(trajectory.full_prefix_token_ids),
            extended_to_step_end=False,
            fixed_no_prefix_train=False,
        )
