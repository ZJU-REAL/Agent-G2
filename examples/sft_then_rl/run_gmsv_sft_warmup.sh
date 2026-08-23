#!/usr/bin/env bash
set -x

NPROC_PER_NODE=${1:-8}
shift || true

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
SFT_JSON=${SFT_JSON:-$ROOT_DIR/sft_then_rl/gmsv_sft_data.json}
SAVE_DIR=${SAVE_DIR:-$ROOT_DIR/checkpoints/ALFWorld_SFT_then_RL/gmsv_sft_warmup}

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    -m sft_then_rl.main_gmsv_sft \
    +data.gmsv_sft_json_path="$SFT_JSON" \
    +data.max_prompt_length=4096 \
    +data.max_response_length=512 \
    +data.use_saved_chat_prompt=true \
    +data.sft_batch_size=128 \
    data.truncation=left \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=Qwen/Qwen2.5-1.5B-Instruct \
    model.enable_gradient_checkpointing=true \
    model.fsdp_config.offload_params=false \
    model.fsdp_config.cpu_offload=false \
    model.strategy=fsdp2 \
    use_remove_padding=false \
    ulysses_sequence_parallel_size=1 \
    trainer.default_local_dir="$SAVE_DIR" \
    trainer.project_name=ALFWorld_SFT_then_RL \
    trainer.experiment_name=gmsv_sft_warmup \
    trainer.total_epochs=1 \
    +trainer.sft_loss_coef=0.5 \
    +trainer.save_freq=-1 \
    trainer.default_hdfs_dir=null \
    "trainer.logger=['console','swanlab']" \
    "$@"
