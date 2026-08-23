# TRAPO for ALFWorld and WebShop

This directory contains the ALFWorld and WebShop reproduction paths for TRAPO.

Method mapping:

- Micro-group sampling: serial rollout groups with sizes `4,2,1,1`.
- Adaptive guidance: each later micro-group uses an expert prefix when the previous pass rate for the same task is less than or equal to its threshold, matching TRAPO Algorithm 1.
- Prefix ratios: default `0,0.2,0.5,1.0`, cut at environment expert action boundaries.
- TrSFT: use `trapo.prefix_sft.loss_type=trapo_clip_low`, whose gradient is `-grad(p) / max(stop_grad(p), alpha)`.
- GRPO details: the TRAPO run scripts disable GRPO std normalization, remove PPO ratio clipping, and use `seq-mean-token-sum-norm` loss aggregation to better match the released TRAPO training script.

The TRAPO run scripts set `trapo.reserve_model_step=false` and `trapo.allow_full_prefix=true`, so `L=1.0` replays the complete expert interaction. If the expert prefix finishes the environment, that micro-group contributes TrSFT supervision only and is omitted from the GRPO rollout batch.
