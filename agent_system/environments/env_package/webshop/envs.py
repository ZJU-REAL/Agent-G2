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

import ray
import gym
import numpy as np
import math
from gmsv.webshop import (
    goal_to_target_key,
    load_expert_target_keys,
    load_webshop_goal_idx_sequence,
    load_webshop_target_key_sequence,
    normalize_target_key,
    target_key_to_string,
)

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)

        env_kwargs = dict(env_kwargs)
        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
        self._target_key_to_goal_idx = {}
        for goal_idx, goal in enumerate(self.env.server.goals):
            target_key = goal_to_target_key(goal)
            self._target_key_to_goal_idx.setdefault(target_key, goal_idx)

    def _add_goal_info(self, info, idx=None):
        session_data = self.env.server.user_sessions.get(self.env.session, {})
        goal = session_data.get("goal")
        if goal is not None:
            target_key = goal_to_target_key(goal)
            info["goal_idx"] = idx
            info["goal"] = goal
            info["instruction_text"] = goal.get("instruction_text")
            info["webshop_target_key"] = target_key
            info["webshop_target_key_text"] = target_key_to_string(target_key)
        return info
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward
        info = self._add_goal_info(info)

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        info = self._add_goal_info(info, idx=idx)
        return obs, info

    def reset_by_target_key(self, target_key):
        """Reset to the local goal that matches the target key."""
        normalized_key = normalize_target_key(target_key)
        idx = self._target_key_to_goal_idx.get(normalized_key)
        if idx is None:
            raise KeyError(f"WebShop worker cannot find target key: {target_key_to_string(normalized_key)}")
        return self.reset(idx)
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = dict(env_kwargs) if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}
        self.train_goal_source = str(self._env_kwargs.pop("train_goal_source", "default"))
        self.train_sample_mode = str(self._env_kwargs.pop("train_sample_mode", "random"))
        self.train_expert_json_path = self._env_kwargs.pop("train_expert_json_path", None)
        self.val_exclude_train_goals = bool(self._env_kwargs.pop("val_exclude_train_goals", False))
        self._train_epoch_order = []
        self._train_epoch_cursor = 0
        self._train_epoch_index = 0

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)
        known_asins = {str(goal.get("asin", "")).upper() for goal in goals}
        train_target_keys = load_expert_target_keys(
            self.train_expert_json_path,
            known_asins=known_asins,
        )
        train_target_key_sequence = load_webshop_target_key_sequence(
            self.train_expert_json_path,
            known_asins=known_asins,
        )
        train_goal_idx_sequence = load_webshop_goal_idx_sequence(
            self.train_expert_json_path,
        )
        all_goal_target_keys = {goal_to_target_key(goal) for goal in goals}
        self._use_target_key_reset = False
        self.goal_target_keys = []

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            goal_idxs = []
            goal_target_keys = []
            seen_val_keys = set()
            for idx in range(len(goals)):
                key = goal_to_target_key(goals[idx])
                if key in seen_val_keys:
                    continue
                seen_val_keys.add(key)
                if self.val_exclude_train_goals and key in train_target_keys:
                    continue
                goal_idxs.append(idx)
                goal_target_keys.append(key)
                if len(goal_idxs) >= 500:
                    break
            if self.val_exclude_train_goals and train_target_keys:
                overlap_count = len({
                    goal_to_target_key(goals[idx]) for idx in goal_idxs
                } & train_target_keys)
                if overlap_count:
                    raise ValueError(f"WebShop validation goals overlap train expert goals: {overlap_count}")
            if len(goal_idxs) < self.env_num:
                raise ValueError(
                    f"WebShop validation goal pool has only {len(goal_idxs)} goals, "
                    f"but env_num={self.env_num}. Decrease data.val_batch_size or use a larger validation split."
                )
            if self.val_exclude_train_goals and len(goal_idxs) < 500:
                raise ValueError(
                    f"WebShop validation goal pool has only {len(goal_idxs)} non-overlapping goals; "
                    "cannot build the requested 500-goal validation set."
                )
            self.goal_idxs = goal_idxs
            self.goal_target_keys = goal_target_keys
            self._use_target_key_reset = bool(self.val_exclude_train_goals)
        else:
            if self.train_goal_source == "expert_json":
                if not train_target_key_sequence:
                    raise ValueError(
                        "env.webshop.train_goal_source=expert_json requires a non-empty "
                        "env.webshop.train_expert_json_path"
                    )
                if train_goal_idx_sequence and len(train_goal_idx_sequence) != len(train_target_key_sequence):
                    raise ValueError(
                        "WebShop train goal_idx sequence and target-key sequence lengths differ: "
                        f"{len(train_goal_idx_sequence)} vs {len(train_target_key_sequence)}"
                    )
                bad_goal_idxs = [
                    goal_idx for goal_idx in train_goal_idx_sequence
                    if goal_idx < 0 or goal_idx >= len(goals)
                ]
                if bad_goal_idxs:
                    raise ValueError(
                        f"WebShop train goal source contains {len(bad_goal_idxs)} out-of-range "
                        f"goal_idx values, examples: {bad_goal_idxs[:3]}"
                    )
                missing_target_keys = [
                    key for key in train_target_key_sequence if key not in all_goal_target_keys
                ]
                if missing_target_keys:
                    examples = ", ".join(target_key_to_string(key) for key in missing_target_keys[:3])
                    raise ValueError(
                        f"WebShop expert train goal source contains {len(missing_target_keys)} "
                        f"keys not found in the local environment goals, examples: {examples}"
                    )
                if train_goal_idx_sequence:
                    self.goal_idxs = train_goal_idx_sequence
                    self.goal_target_keys = []
                    self._use_target_key_reset = False
                else:
                    self.goal_idxs = []
                    self.goal_target_keys = train_target_key_sequence
                    self._use_target_key_reset = True
                goal_pool_size = len(self.goal_target_keys) if self._use_target_key_reset else len(self.goal_idxs)
                if goal_pool_size < self.env_num:
                    raise ValueError(
                        f"WebShop expert train goal pool has only {goal_pool_size} goals, "
                        f"but env_num={self.env_num}."
                    )
            else:
                self.goal_idxs = list(range(500, len(goals)))
            
        goal_count = len(self.goal_target_keys) if self._use_target_key_reset else len(self.goal_idxs)
        print(f"WebShop {'train' if self.is_train else 'validation'} goal count: {goal_count}")
        if self.is_train:
            if self.train_sample_mode not in {"random", "epoch_shuffle", "sequence"}:
                raise ValueError(
                    "env.webshop.train_sample_mode must be 'random', 'epoch_shuffle', or "
                    f"'sequence', got {self.train_sample_mode!r}"
                )
            print(f"WebShop train sample mode: {self.train_sample_mode}")
            if self.train_sample_mode in {"epoch_shuffle", "sequence"}:
                print(f"WebShop train batches per no-repeat cycle: {math.ceil(goal_count / self.env_num)}")

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def step_selected(self, indices: list[int], actions: list[str]):
        if len(indices) != len(actions):
            raise ValueError(f"Expected equal indices/actions length, got {len(indices)} and {len(actions)}")

        futures = []
        for env_idx, action in zip(indices, actions):
            futures.append(self._workers[env_idx].step.remote(action))

        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def _reset_train_epoch_order(self):
        pool_size = len(self.goal_target_keys) if self._use_target_key_reset else len(self.goal_idxs)
        if pool_size < self.env_num:
            raise ValueError(
                f"WebShop train goal pool has only {pool_size} goals, but env_num={self.env_num}."
            )
        if self.train_sample_mode == "sequence":
            self._train_epoch_order = list(range(pool_size))
        else:
            self._train_epoch_order = self._rng.permutation(pool_size).tolist()
        self._train_epoch_cursor = 0
        self._train_epoch_index += 1

    def _sample_train_positions(self):
        if self.train_sample_mode not in {"epoch_shuffle", "sequence"}:
            pool_size = len(self.goal_target_keys) if self._use_target_key_reset else len(self.goal_idxs)
            return self._rng.choice(pool_size, size=self.env_num, replace=False).tolist()

        if not self._train_epoch_order:
            self._reset_train_epoch_order()

        positions = []
        while len(positions) < self.env_num:
            if self._train_epoch_cursor >= len(self._train_epoch_order):
                self._reset_train_epoch_order()
            take = min(self.env_num - len(positions), len(self._train_epoch_order) - self._train_epoch_cursor)
            positions.extend(
                self._train_epoch_order[
                    self._train_epoch_cursor : self._train_epoch_cursor + take
                ]
            )
            self._train_epoch_cursor += take
        return positions

    def reset(self):
        if self._use_target_key_reset:
            if self.is_train:
                sampled_positions = self._sample_train_positions()
            else:
                sampled_positions = self._rng.choice(len(self.goal_target_keys), size=self.env_num, replace=False).tolist()
            target_keys = [self.goal_target_keys[int(pos)] for pos in sampled_positions]
            reset_items = [
                target_key
                for target_key in target_keys
                for _ in range(self.group_n)
            ]
        else:
            if self.is_train:
                sampled_positions = self._sample_train_positions()
                idx = [self.goal_idxs[int(pos)] for pos in sampled_positions]
            else:
                idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False).tolist()
            reset_items = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, reset_item in zip(self._workers, reset_items):
            if self._use_target_key_reset:
                future = worker.reset_by_target_key.remote(reset_item)
            else:
                future = worker.reset.remote(reset_item)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return
        if not hasattr(self, '_workers'):
            self._closed = True
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
