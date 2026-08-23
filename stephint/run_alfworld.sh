set -x
ENGINE=${1:-vllm}
shift || true
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
ALFWORLD_EXPERT_JSON=$ROOT_DIR/sft_data/alfworld_sft_data.json
num_cpus_per_env_worker=0.1

train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-128}
group_size=${GROUP_SIZE:-12}

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m stephint.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=false \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=false \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=20 \
    env.history_length=20 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +step_hint.enable=true \
    +step_hint.expert_json_path=$ALFWORLD_EXPERT_JSON \
    +step_hint.apply_on_train=true \
    +step_hint.apply_on_validation=false \
    +step_hint.partition_count=4 \
    +step_hint.khint=2 \
    +step_hint.kunhint=5 \
    +step_hint.include_reference=true \
    +step_hint.protect_negative_advantage=true \
    +step_hint.strict_group_size=true \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,swanlab]" \
    trainer.project_name=ALFWorld_StepHint \
    trainer.experiment_name=alfworld_stephint_1.5B \
    trainer.default_local_dir=checkpoints/ALFWorld_StepHint/alfworld_stephint_1.5B \
    trainer.train_rollout_detail_data_dir=null \
    trainer.val_rollout_detail_data_dir=null \
    trainer.rollout_detail_dump_freq=0 \
    trainer.rollout_detail_dump_full=false \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=300 \
    trainer.val_before_train=True $@
