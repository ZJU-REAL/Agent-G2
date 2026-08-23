# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Any, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.prefix_runtime = None
        self.prefix_runtime_name = None
        self.gmsv_runtime = None
        self.binary_search_runtime = None
        self.enumerate_hint_runtime = None
        self.cosine_hint_runtime = None
        self.liner_hint_runtime = None
        self.fix_step_hint_runtime = None
        self.fix_acc_hint_runtime = None
        self.trapo_runtime = None
        self.latest_prefix_sft_batch = None
        self._pending_prefix_sft_batches = []
        self._last_prefix_sft_records = None
        self._last_prefix_sft_traj_uid = None
        self._last_prefix_detail_records = None
        gmsv_enabled = bool(self.config.get("gmsv", {}).get("enable", False))
        binary_search_enabled = bool(self.config.get("binary_search", {}).get("enable", False))
        enumerate_hint_enabled = bool(self.config.get("enumerate_hint", {}).get("enable", False))
        cosine_hint_enabled = bool(self.config.get("cosine_hint", {}).get("enable", False))
        liner_hint_enabled = bool(self.config.get("liner_hint", {}).get("enable", False))
        fix_step_hint_enabled = bool(self.config.get("fix_step_hint", {}).get("enable", False))
        fix_acc_hint_enabled = bool(self.config.get("fix_acc_hint", {}).get("enable", False))
        trapo_enabled = bool(self.config.get("trapo", {}).get("enable", False))
        enabled_prefix_runtimes = [
            name
            for name, enabled in (
                ("gmsv", gmsv_enabled),
                ("binary_search", binary_search_enabled),
                ("enumerate_hint", enumerate_hint_enabled),
                ("cosine_hint", cosine_hint_enabled),
                ("liner_hint", liner_hint_enabled),
                ("fix_step_hint", fix_step_hint_enabled),
                ("fix_acc_hint", fix_acc_hint_enabled),
                ("trapo", trapo_enabled),
            )
            if enabled
        ]
        if len(enabled_prefix_runtimes) > 1:
            raise ValueError(f"Only one prefix runtime can be enabled, got: {enabled_prefix_runtimes}.")
        if gmsv_enabled:
            from gmsv import build_gmsv_runtime
            self.gmsv_runtime = build_gmsv_runtime(tokenizer=self.tokenizer, config=self.config)
            self.prefix_runtime = self.gmsv_runtime
            self.prefix_runtime_name = "gmsv"
        elif binary_search_enabled:
            from binary_search import build_binary_search_runtime
            self.binary_search_runtime = build_binary_search_runtime(tokenizer=self.tokenizer, config=self.config)
            self.prefix_runtime = self.binary_search_runtime
            self.prefix_runtime_name = "binary_search"
        elif enumerate_hint_enabled:
            from enumerate_hint import build_enumerate_hint_runtime
            self.enumerate_hint_runtime = build_enumerate_hint_runtime(tokenizer=self.tokenizer, config=self.config)
            self.prefix_runtime = self.enumerate_hint_runtime
            self.prefix_runtime_name = "enumerate_hint"
        elif cosine_hint_enabled:
            from cosine_hint import build_cosine_hint_runtime
            self.cosine_hint_runtime = build_cosine_hint_runtime(tokenizer=self.tokenizer, config=self.config)
            self.prefix_runtime = self.cosine_hint_runtime
            self.prefix_runtime_name = "cosine_hint"
        elif liner_hint_enabled:
            from liner_hint import build_liner_hint_runtime
            self.liner_hint_runtime = build_liner_hint_runtime(tokenizer=self.tokenizer, config=self.config)
            self.prefix_runtime = self.liner_hint_runtime
            self.prefix_runtime_name = "liner_hint"
        elif fix_step_hint_enabled:
            from fix_step_hint import build_fix_step_hint_runtime
            self.fix_step_hint_runtime = build_fix_step_hint_runtime(tokenizer=self.tokenizer, config=self.config)
            self.prefix_runtime = self.fix_step_hint_runtime
            self.prefix_runtime_name = "fix_step_hint"
        elif fix_acc_hint_enabled:
            from fix_acc_hint import build_fix_acc_hint_runtime
            self.fix_acc_hint_runtime = build_fix_acc_hint_runtime(tokenizer=self.tokenizer, config=self.config)
            self.prefix_runtime = self.fix_acc_hint_runtime
            self.prefix_runtime_name = "fix_acc_hint"
        elif trapo_enabled:
            from trapo import build_trapo_runtime
            self.trapo_runtime = build_trapo_runtime(tokenizer=self.tokenizer, config=self.config)
            self.prefix_runtime = self.trapo_runtime
            self.prefix_runtime_name = "trapo"

    def _jsonify(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._jsonify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonify(v) for v in value]
        return value

    def _decode_ids(self, token_ids, skip_special_tokens: bool = False) -> str:
        token_ids = self._jsonify(token_ids)
        if token_ids is None:
            return ""
        if not isinstance(token_ids, list):
            token_ids = [token_ids]
        token_ids = [int(token_id) for token_id in token_ids if token_id is not None]
        if not token_ids:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _valid_response_length(self, step_data: Dict[str, Any]) -> int | None:
        responses = step_data.get("responses")
        if responses is None:
            return None
        response_ids = self._jsonify(responses)
        response_length = len(response_ids) if isinstance(response_ids, list) else 0
        attention_mask = step_data.get("attention_mask")
        prompts = step_data.get("prompts")
        if attention_mask is None or prompts is None:
            return response_length
        attention_values = self._jsonify(attention_mask)
        prompt_values = self._jsonify(prompts)
        if not isinstance(attention_values, list) or not isinstance(prompt_values, list):
            return response_length
        prompt_length = len(prompt_values)
        response_mask = attention_values[prompt_length:prompt_length + response_length]
        return int(sum(response_mask))

    def _decode_step_response(self, step_data: Dict[str, Any]) -> str:
        response_ids = self._jsonify(step_data.get("responses"))
        if response_ids is None:
            return ""
        valid_response_length = self._valid_response_length(step_data)
        if valid_response_length is not None:
            response_ids = response_ids[:valid_response_length]
        return self._decode_ids(response_ids, skip_special_tokens=True)

    def _extract_xml_tag(self, text: str, tag: str) -> str:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _should_dump_rollout_detail(self, dump_dir: str | None, global_step: int | None) -> bool:
        if not dump_dir or global_step is None or global_step <= 0:
            return False
        dump_freq = int(self.config.trainer.get("rollout_detail_dump_freq", 0) or 0)
        return dump_freq > 0 and global_step % dump_freq == 0

    def _should_dump_full_rollout_detail(self, dump_dir: str | None, global_step: int | None) -> bool:
        if not bool(self.config.trainer.get("rollout_detail_dump_full", True)):
            return False
        return self._should_dump_rollout_detail(dump_dir=dump_dir, global_step=global_step)

    def _should_dump_sft_pairs(self, dump_dir: str | None, global_step: int | None, is_train: bool) -> bool:
        return (
            is_train
            and self._prefix_sft_enabled(is_train=True)
            and self._should_dump_rollout_detail(dump_dir=dump_dir, global_step=global_step)
        )

    def _rollout_detail_step(self, step_data: Dict[str, Any], step_idx: int, cumulative_reward: float) -> Dict[str, Any]:
        raw_model_output = self._decode_step_response(step_data)
        prompt_ids = step_data.get("raw_prompt_ids", step_data.get("prompts", step_data.get("input_ids")))
        info = self._jsonify(step_data.get("_rollout_detail_info", {})) or {}
        reward = self._jsonify(step_data.get("rewards", step_data.get("_rollout_detail_reward", 0.0)))
        if isinstance(reward, list):
            reward = reward[0] if reward else 0.0

        return {
            "step_id": step_idx,
            "phase": "model",
            "source": "policy",
            "prompt": self._decode_ids(prompt_ids, skip_special_tokens=False),
            "observation": self._jsonify(step_data.get("_rollout_detail_observation")),
            "prompt_observation": self._jsonify(step_data.get("_rollout_detail_prompt_observation")),
            "anchor_observation": self._jsonify(step_data.get("anchor_obs")),
            "available_actions": self._jsonify(step_data.get("_rollout_detail_available_actions")),
            "raw_model_output": raw_model_output,
            "think": self._extract_xml_tag(raw_model_output, "think"),
            "action": self._extract_xml_tag(raw_model_output, "action"),
            "is_action_valid": bool(self._jsonify(step_data.get("is_action_valid", True))),
            "env_reward": float(reward),
            "cumulative_reward": float(cumulative_reward),
            "done": bool(self._jsonify(step_data.get("_rollout_detail_done", False))),
            "won": bool(info.get("won", False)),
            "task_score": info.get("task_score"),
            "info": info,
            "next_observation": self._jsonify(step_data.get("_rollout_detail_next_observation")),
            "prompt_token_len": len(self._jsonify(prompt_ids) or []),
            "response_token_len": self._valid_response_length(step_data),
        }

    def _build_rollout_details_payload(
        self,
        *,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
        global_step: int | None,
        is_train: bool,
        prefix_detail_records: List[List[Dict[str, Any]]] | None = None,
    ) -> Dict[str, Any]:
        group_size = int(self.config.env.rollout.n) if is_train and self.config.env.rollout.n > 0 else 1
        trajectories = []

        for traj_idx, trajectory_steps in enumerate(total_batch_list):
            prefix_steps = []
            if prefix_detail_records is not None and traj_idx < len(prefix_detail_records):
                prefix_steps = [self._jsonify(step) for step in (prefix_detail_records[traj_idx] or [])]
            active_steps = [
                step_data for step_data in trajectory_steps
                if bool(self._jsonify(step_data.get("active_masks", True)))
            ]
            if not active_steps and not prefix_steps:
                continue

            serialized_steps = []
            cumulative_reward = 0.0
            for prefix_step_idx, prefix_step in enumerate(prefix_steps):
                prefix_step["step_id"] = prefix_step_idx
                cumulative_reward = float(prefix_step.get("cumulative_reward", cumulative_reward + float(prefix_step.get("env_reward", 0.0) or 0.0)))
                serialized_steps.append(prefix_step)

            for model_step_offset, step_data in enumerate(active_steps):
                step_reward = self._jsonify(step_data.get("rewards", 0.0))
                if isinstance(step_reward, list):
                    step_reward = step_reward[0] if step_reward else 0.0
                cumulative_reward += float(step_reward)
                serialized_steps.append(self._rollout_detail_step(step_data, len(serialized_steps), cumulative_reward))

            first_step = active_steps[0] if active_steps else {}
            last_step = active_steps[-1] if active_steps else {}
            last_serialized_step = serialized_steps[-1] if serialized_steps else {}
            success_values = {
                key: self._jsonify(value[traj_idx]) if traj_idx < len(value) else None
                for key, value in success.items()
            }
            trial_id = (
                self._jsonify(first_step.get("prefix_probe_trial_id"))
                or self._jsonify(first_step.get("gmsv_trial_id"))
                or self._jsonify(last_serialized_step.get("trial_id"))
                or ""
            )
            trajectories.append({
                "uid": str(first_step.get("uid", "")),
                "traj_uid": str(traj_uid[traj_idx]),
                "group_index": traj_idx // group_size,
                "sample_index": traj_idx % group_size,
                "trial_id": str(trial_id),
                "prefix_probe_ratio": self._jsonify(first_step.get("prefix_probe_ratio")),
                "prefix_probe_requested_step_count": self._jsonify(first_step.get("prefix_probe_requested_step_count")),
                "prefix_probe_prefix_step_count": self._jsonify(first_step.get("prefix_probe_prefix_step_count")),
                "prefix_probe_total_step_count": self._jsonify(first_step.get("prefix_probe_total_step_count")),
                "task": self._jsonify(first_step.get("_rollout_detail_task")),
                "episode_reward": float(episode_rewards[traj_idx]),
                "episode_length": float(episode_lengths[traj_idx]),
                "success": success_values,
                "tool_callings": float(tool_callings[traj_idx]),
                "final_info": self._jsonify(last_step.get("_rollout_detail_info", {})),
                "steps": serialized_steps,
            })

        return {
            "global_step": int(global_step) if global_step is not None else None,
            "split": "train" if is_train else "val",
            "env_name": self.config.env.env_name,
            "train_batch_size": int(self.config.data.train_batch_size),
            "val_batch_size": int(self.config.data.val_batch_size) if self.config.data.val_batch_size is not None else None,
            "group_size": group_size,
            "max_steps": int(self.config.env.max_steps),
            "num_trajectories": len(trajectories),
            "trajectories": trajectories,
        }

    def _dump_rollout_details_payload(
        self,
        *,
        payload: Optional[Dict[str, Any]],
        dump_dir: str | None,
        global_step: int | None,
    ) -> None:
        if payload is None or not self._should_dump_full_rollout_detail(dump_dir=dump_dir, global_step=global_step):
            return
        os.makedirs(dump_dir, exist_ok=True)
        output_path = os.path.join(dump_dir, f"global_step_{global_step}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Dumped detailed rollout trajectories to {output_path}")

    def _build_sft_pairs_payload(self, rollout_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not rollout_payload or rollout_payload.get("split") != "train":
            return None
        if not self._prefix_sft_enabled(is_train=True):
            return None

        sft_pairs = []
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        append_eos = bool(self._prefix_sft_cfg().get("append_eos", True))

        for trajectory in rollout_payload.get("trajectories", []):
            prefix_steps = [
                step
                for step in trajectory.get("steps", [])
                if step.get("phase") == "prefix"
                and step.get("source") == "expert"
                and str(step.get("action", "")).strip()
                and str(step.get("prompt", ""))
            ]
            if not prefix_steps:
                continue

            step_weight = 1.0 / float(len(prefix_steps))
            for sft_step_index, step in enumerate(prefix_steps):
                prompt = str(step.get("prompt", ""))
                think = str(step.get("think", "")).strip()
                action = str(step.get("action", "")).strip()
                response = action
                if think:
                    response = f"<think>{think}</think>\n<action>{action}</action>"

                chat = np.array([{
                    "content": prompt,
                    "role": "user",
                }])
                chat_prompt = self.tokenizer.apply_chat_template(
                    chat,
                    add_generation_prompt=True,
                    tokenize=False,
                    **apply_chat_template_kwargs,
                )

                sft_pairs.append({
                    "global_step": rollout_payload.get("global_step"),
                    "env_name": rollout_payload.get("env_name"),
                    "traj_uid": trajectory.get("traj_uid"),
                    "uid": trajectory.get("uid"),
                    "trial_id": trajectory.get("trial_id"),
                    "group_index": trajectory.get("group_index"),
                    "sample_index": trajectory.get("sample_index"),
                    "sft_step_index": sft_step_index,
                    "step_id": step.get("step_id"),
                    "prefix_step_count": step.get("prefix_step_count"),
                    "total_step_count": step.get("total_step_count"),
                    "prompt": prompt,
                    "chat_prompt": chat_prompt,
                    "response": response,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    "think": think,
                    "action": action,
                    "loss_weight": step_weight,
                    "append_eos": append_eos,
                    "observation": step.get("observation"),
                    "anchor_observation": step.get("anchor_observation"),
                    "available_actions": step.get("available_actions"),
                    "env_reward": step.get("env_reward"),
                    "cumulative_reward": step.get("cumulative_reward"),
                    "done": step.get("done"),
                    "won": step.get("won"),
                    "task_score": step.get("task_score"),
                    "info": step.get("info"),
                    "next_observation": step.get("next_observation"),
                    "phase": step.get("phase"),
                    "source": step.get("source"),
                })

        return {
            "global_step": rollout_payload.get("global_step"),
            "split": rollout_payload.get("split"),
            "env_name": rollout_payload.get("env_name"),
            "num_pairs": len(sft_pairs),
            "pairs": sft_pairs,
        }

    def _dump_sft_pairs_payload(
        self,
        *,
        rollout_payload: Optional[Dict[str, Any]],
        dump_dir: str | None,
        global_step: int | None,
    ) -> None:
        if not self._should_dump_rollout_detail(dump_dir=dump_dir, global_step=global_step):
            return
        payload = self._build_sft_pairs_payload(rollout_payload)
        if payload is None or payload["num_pairs"] <= 0:
            return

        sft_pairs_dir = os.path.join(dump_dir, "sft_pairs")
        os.makedirs(sft_pairs_dir, exist_ok=True)
        output_path = os.path.join(sft_pairs_dir, f"global_step_{global_step}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Dumped prefix SFT training pairs to {output_path}")

    def _prefix_runtime_cfg(self):
        if self.prefix_runtime_name is None:
            return {}
        return self.config.get(self.prefix_runtime_name, {})

    def _prefix_sft_cfg(self):
        return self._prefix_runtime_cfg().get("prefix_sft", {})

    def _prefix_sft_enabled(self, is_train: bool) -> bool:
        if not is_train:
            return False
        return bool(self._prefix_sft_cfg().get("enable", False))

    def _has_policy_rollout_steps(self, total_batch_list: List[List[Dict]]) -> bool:
        for trajectory_steps in total_batch_list:
            for data in trajectory_steps:
                if bool(data.get("active_masks", False)):
                    return True
        return False

    def _build_prefix_sft_batch(
        self,
        gen_batch: DataProto,
        prefix_sft_records: List[List[Dict[str, Any]]] | None,
        traj_uid: np.ndarray | None,
        keep_traj_uids: set[str] | None = None,
    ) -> DataProto | None:
        if not prefix_sft_records or traj_uid is None:
            return None

        sft_cfg = self._prefix_sft_cfg()
        append_eos = bool(sft_cfg.get("append_eos", True))
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        max_prompt_length = self.config.data.max_prompt_length
        max_response_length = self.config.data.max_response_length

        samples = []
        for env_idx, env_records in enumerate(prefix_sft_records):
            if env_idx >= len(traj_uid):
                continue
            traj_key = str(traj_uid[env_idx])
            if keep_traj_uids is not None and traj_key not in keep_traj_uids:
                continue
            valid_records = [
                record for record in env_records
                if str(record.get("action", "")).strip() and str(record.get("obs_text", ""))
            ]
            if not valid_records:
                continue
            step_weight = 1.0 / float(len(valid_records))

            for record in valid_records:
                action = str(record.get("action", "")).strip()
                think = str(record.get("think", "")).strip()
                obs_text = str(record.get("obs_text", ""))

                chat = np.array([{
                    "content": obs_text,
                    "role": "user",
                }])
                prompt = self.tokenizer.apply_chat_template(
                    chat,
                    add_generation_prompt=True,
                    tokenize=False,
                    **apply_chat_template_kwargs,
                )
                prompt_ids, prompt_attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=prompt,
                    tokenizer=self.tokenizer,
                    max_length=max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.config.data.truncation,
                )
                prompt_position_ids = compute_position_id_with_mask(prompt_attention_mask)

                response_text = action
                if think:
                    response_text = f"<think>{think}</think>\n<action>{action}</action>"
                response_ids_list = self.tokenizer.encode(response_text, add_special_tokens=False)
                if append_eos and self.tokenizer.eos_token_id is not None:
                    response_ids_list.append(self.tokenizer.eos_token_id)
                if not response_ids_list:
                    continue

                response_ids = torch.tensor(response_ids_list, dtype=torch.long).unsqueeze(0)
                response_attention_mask = torch.ones_like(response_ids)
                response_ids, response_attention_mask = verl_F.postprocess_data(
                    input_ids=response_ids,
                    attention_mask=response_attention_mask,
                    max_length=max_response_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=False,
                    truncation="right",
                )

                response_position_delta = torch.arange(1, max_response_length + 1).unsqueeze(0)
                response_position_ids = prompt_position_ids[:, -1:] + response_position_delta

                input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
                attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
                position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
                prompt_loss_mask = torch.zeros_like(prompt_attention_mask)
                loss_mask = torch.cat([prompt_loss_mask, response_attention_mask], dim=-1)

                samples.append({
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0],
                    "position_ids": position_ids[0],
                    "responses": response_ids[0],
                    "loss_mask": loss_mask[0],
                    "prefix_sft_mask": response_attention_mask[0].bool(),
                    "prefix_sft_step_weight": torch.tensor(step_weight, dtype=torch.float32),
                    "old_log_probs": torch.zeros(max_response_length, dtype=torch.float32),
                    "advantages": torch.zeros(max_response_length, dtype=torch.float32),
                })

        if not samples:
            return None

        return DataProto.from_single_dict(data=collate_fn(samples), meta_info=gen_batch.meta_info)

    def _append_prefix_sft_batch(
        self,
        gen_batch: DataProto,
        is_train: bool,
        keep_traj_uids: np.ndarray | None = None,
    ) -> None:
        if not self._prefix_sft_enabled(is_train=is_train):
            return

        keep_set = None
        if keep_traj_uids is not None:
            keep_set = {str(uid) for uid in keep_traj_uids}

        prefix_sft_batch = self._build_prefix_sft_batch(
            gen_batch=gen_batch,
            prefix_sft_records=self._last_prefix_sft_records,
            traj_uid=self._last_prefix_sft_traj_uid,
            keep_traj_uids=keep_set,
        )
        if prefix_sft_batch is not None:
            self._pending_prefix_sft_batches.append(prefix_sft_batch)

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    data = {
                        key: value
                        for key, value in data.items()
                        if not str(key).startswith("_rollout_detail_")
                    }
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)

        if not effective_batch:
            raise RuntimeError(
                "No policy-generated rollout steps were collected. This can happen when all trajectories "
                "are completed by expert prefix replay; reduce full-prefix sampling probability or disable "
                "gmsv.allow_full_prefix."
            )

        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            prefix_runtime_override=None,
            collect_rollout_details: bool = False,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        prefix_meta: Dict[str, np.ndarray] = {}
        prefix_rewards = np.zeros(batch_size, dtype=np.float32)
        prefix_lengths = np.zeros(batch_size, dtype=np.float32)
        prefix_dones = np.zeros(batch_size, dtype=bool)
        collect_prefix_sft = self._prefix_sft_enabled(is_train)
        active_prefix_runtime = prefix_runtime_override if prefix_runtime_override is not None else self.prefix_runtime
        if (
            active_prefix_runtime is not None
            and active_prefix_runtime.should_apply(env_name=self.config.env.env_name, is_train=is_train)
            and hasattr(envs, "replay_prefix_plans")
        ):
            prefix_cfg = getattr(active_prefix_runtime, "prefix_cfg", None)
            if prefix_cfg is None:
                prefix_cfg = self._prefix_runtime_cfg()
            prefix_plans = active_prefix_runtime.build_prefix_plans(
                infos=infos,
                is_train=is_train,
                group_size=self.config.env.rollout.n if is_train and self.config.env.rollout.n > 0 else 1,
                share_within_group=bool(prefix_cfg.get("share_prefix_within_group", True)),
            )
            prefix_meta = active_prefix_runtime.build_batch_metadata(prefix_plans)
            obs, prefix_rewards, prefix_lengths, prefix_dones = envs.replay_prefix_plans(
                prefix_plans,
                collect_sft=collect_prefix_sft,
                collect_detail=collect_rollout_details,
            )
            self._last_prefix_detail_records = getattr(envs, "prefix_detail_records", None) if collect_rollout_details else None

        full_prefix_done = np.asarray(
            prefix_meta.get("gmsv_full_prefix", np.zeros(batch_size, dtype=bool)),
            dtype=bool,
        )
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        remaining_step_budget = np.maximum(
            int(self.config.env.max_steps) - prefix_lengths.astype(np.int32),
            0,
        ).astype(np.int32)
        is_done = np.logical_or(prefix_dones.copy(), remaining_step_budget <= 0)
        is_done = np.logical_or(is_done, full_prefix_done)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        self._last_prefix_sft_records = getattr(envs, "prefix_sft_records", None) if collect_prefix_sft else None
        self._last_prefix_sft_traj_uid = traj_uid if collect_prefix_sft else None
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = prefix_lengths.copy()
        episode_rewards = prefix_rewards.copy()
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            if not active_masks.any():
                break

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            if prefix_meta:
                batch.non_tensor_batch.update(prefix_meta)

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            current_infos = getattr(envs, "last_infos", None)
            current_tasks = getattr(envs, "tasks", None)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                current_info = current_infos[i] if isinstance(current_infos, list) and i < len(current_infos) else {}
                if not isinstance(current_info, dict):
                    current_info = {}
                current_available_actions = current_info.get("available_actions", current_info.get("admissible_commands"))
                current_task = current_tasks[i] if isinstance(current_tasks, list) and i < len(current_tasks) else None
                current_observation = None
                current_prompt_observation = None
                if obs.get("text", None) is not None:
                    current_prompt_observation = obs["text"][i]
                    current_observation = obs["text"][i]
                if obs.get("anchor", None) is not None:
                    current_observation = obs["anchor"][i]

                batch_list[i]["_rollout_detail_step"] = _step
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

            # Update done states
            remaining_step_budget[active_masks] -= 1
            is_done = np.logical_or(is_done, dones)
            is_done = np.logical_or(is_done, remaining_step_budget <= 0)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            prefix_runtime_override=None,
            collect_rollout_details: bool = False,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
                prefix_runtime_override=prefix_runtime_override,
                collect_rollout_details=collect_rollout_details,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            self._append_prefix_sft_batch(gen_batch=gen_batch, is_train=is_train, keep_traj_uids=traj_uid)
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            prefix_runtime_override=None,
            global_step: int | None = None,
            rollout_detail_dump_dir: str | None = None,
            collect_rollout_details: bool = False,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        self.latest_prefix_sft_batch = None
        self._pending_prefix_sft_batches = []
        self._last_prefix_sft_records = None
        self._last_prefix_sft_traj_uid = None
        self._last_prefix_detail_records = None
        should_dump_full_rollout_details = self._should_dump_full_rollout_detail(
            dump_dir=rollout_detail_dump_dir,
            global_step=global_step,
        )
        should_dump_sft_pairs = self._should_dump_sft_pairs(
            dump_dir=rollout_detail_dump_dir,
            global_step=global_step,
            is_train=is_train,
        )
        should_collect_rollout_details = bool(collect_rollout_details) or should_dump_full_rollout_details or should_dump_sft_pairs

        if (
            is_train
            and prefix_runtime_override is None
            and self.prefix_runtime_name == "binary_search"
            and self.binary_search_runtime is not None
        ):
            from binary_search import run_binary_search_multi_turn_loop

            return run_binary_search_multi_turn_loop(
                collector=self,
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                global_step=global_step,
                rollout_detail_dump_dir=rollout_detail_dump_dir,
                collect_rollout_details=collect_rollout_details,
            )

        if (
            is_train
            and prefix_runtime_override is None
            and self.prefix_runtime_name == "enumerate_hint"
            and self.enumerate_hint_runtime is not None
        ):
            from enumerate_hint import run_enumerate_hint_multi_turn_loop

            return run_enumerate_hint_multi_turn_loop(
                collector=self,
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                global_step=global_step,
                rollout_detail_dump_dir=rollout_detail_dump_dir,
                collect_rollout_details=collect_rollout_details,
            )

        if (
            is_train
            and prefix_runtime_override is None
            and self.prefix_runtime_name == "trapo"
            and self.trapo_runtime is not None
        ):
            from trapo import run_trapo_multi_turn_loop

            return run_trapo_multi_turn_loop(
                collector=self,
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                global_step=global_step,
                rollout_detail_dump_dir=rollout_detail_dump_dir,
                collect_rollout_details=collect_rollout_details,
            )

        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
                prefix_runtime_override=prefix_runtime_override,
                collect_rollout_details=False,
            )
        else:
            # Vanilla Sampling   
            max_empty_retries = int(self._prefix_runtime_cfg().get("empty_policy_rollout_retry_limit", 3)) if is_train else 0
            for empty_retry_idx in range(max_empty_retries + 1):
                total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                    self.vanilla_multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                    is_train=is_train,
                    prefix_runtime_override=prefix_runtime_override,
                    collect_rollout_details=should_collect_rollout_details,
                )
                if self._has_policy_rollout_steps(total_batch_list):
                    break
                if empty_retry_idx < max_empty_retries:
                    print(
                        "No policy rollout steps collected after prefix replay; "
                        f"retrying rollout batch ({empty_retry_idx + 1}/{max_empty_retries})."
                    )
            self._append_prefix_sft_batch(gen_batch=gen_batch, is_train=is_train, keep_traj_uids=total_traj_uid)
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        
        rollout_details_payload = None
        if should_collect_rollout_details:
            prefix_detail_records = self._last_prefix_detail_records
            if self.config.algorithm.filter_groups.enable and is_train:
                prefix_detail_records = None
            rollout_details_payload = self._build_rollout_details_payload(
                total_batch_list=total_batch_list,
                episode_rewards=total_episode_rewards,
                episode_lengths=total_episode_lengths,
                success=total_success,
                traj_uid=total_traj_uid,
                tool_callings=totoal_tool_callings,
                global_step=global_step,
                is_train=is_train,
                prefix_detail_records=prefix_detail_records,
            )
        self._dump_rollout_details_payload(
            payload=rollout_details_payload,
            dump_dir=rollout_detail_dump_dir,
            global_step=global_step,
        )
        self._dump_sft_pairs_payload(
            rollout_payload=rollout_details_payload,
            dump_dir=rollout_detail_dump_dir,
            global_step=global_step,
        )

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        if collect_rollout_details and rollout_details_payload is not None:
            gen_batch_output.meta_info["rollout_details"] = rollout_details_payload
        if self._pending_prefix_sft_batches:
            self.latest_prefix_sft_batch = DataProto.concat(self._pending_prefix_sft_batches)
        
        return gen_batch_output
