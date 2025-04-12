#!/bin/bash

# data
# POSTFIX=spatial
POSTFIX=goal
# POSTFIX=10
DATA_NAME=libero_${POSTFIX}
DATA_ROOT=${DATA_NAME}_no_noops

CUDA_VISIBLE_DEVICES=3 python vla-scripts/merge.py \
  --vla_path "MODEL/openvla-7b-finetuned-libero-${POSTFIX}" \
  --data_root_dir ./data/modified_libero_rlds \
  --dataset_name ${DATA_ROOT} \
  --run_root_dir checkpoints/libero_goal_no_noops/root/ppo+libero_goal_no_noops+rb10+tb16+lr-2e-05+vlr-0.0005+s-1+lora \
  --adapter_tmp_dir checkpoints/libero_goal_no_noops/root/ppo+libero_goal_no_noops+rb10+tb16+lr-2e-05+vlr-0.0005+s-1+lora/step_20 \
  --lora_rank 32 \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project openvla \
  --wandb_entity  openvla_cvpr \
  --save_steps 10000 \
  --max_steps 100000 \
  --load_model True \
  --copy_needed_files True

  # --just_copy_needed_files False \

  # --adapter_tmp_dir checkpoints/libero_goal_no_noops/root/ppo+libero_goal_no_noops+rb10+tb16+lr-2e-05+vlr-0.0005+s-1+lora \


