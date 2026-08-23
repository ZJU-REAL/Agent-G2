#!/usr/bin/env bash
set -x

ENGINE=${1:-vllm}
shift || true

SFT_CKPT=${SFT_CKPT:-}
if [ "$#" -gt 0 ] && [ -d "$1" ]; then
    SFT_CKPT=$1
    shift || true
fi

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
SFT_SAVE_DIR=${SFT_SAVE_DIR:-$ROOT_DIR/checkpoints/ALFWorld_SFT_then_RL/gmsv_sft_warmup}

if [ -z "$SFT_CKPT" ]; then
    SFT_CKPT=$(ls -d "$SFT_SAVE_DIR"/global_step_* 2>/dev/null | sort -V | tail -1)
fi

if [ -z "$SFT_CKPT" ] || [ ! -d "$SFT_CKPT" ]; then
    echo "Could not find SFT checkpoint. Pass it as the second argument or set SFT_SAVE_DIR." >&2
    exit 1
fi

bash "$ROOT_DIR/examples/grpo_trainer/run_alfworld.sh" "$ENGINE" \
    trainer.project_name=ALFWorld_SFT_then_RL \
    trainer.experiment_name=alfworld_sft_then_grpo_300step \
    trainer.default_local_dir=checkpoints/ALFWorld_SFT_then_RL/alfworld_sft_then_grpo_300step \
    actor_rollout_ref.model.path="$SFT_CKPT" \
    env.history_length=20 \
    trainer.total_training_steps=300 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.train_rollout_detail_data_dir=rollout_details/ALFWorld_SFT_then_RL/alfworld_sft_then_grpo_300step/train \
    trainer.val_rollout_detail_data_dir=rollout_details/ALFWorld_SFT_then_RL/alfworld_sft_then_grpo_300step/val \
    trainer.rollout_detail_dump_freq=30 \
    "trainer.logger=['console','swanlab']" \
    "$@"
