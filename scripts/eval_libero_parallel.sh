#!/bin/bash

# Configurable parameters
TOTAL_TASKS=10  # Total number of tasks in the suite (adjust based on your task suite)
NUM_GPUS=4      # Number of GPUs available (adjust based on your system)
TASKS_PER_GPU=$(($TOTAL_TASKS / $NUM_GPUS))

export MUJOCO_GL=osmesa
export DATA_ROOT=libero_10_no_noops
export DATA_NAME=libero_10
CKPT_PATH="checkpoints_copy/${DATA_ROOT}/root/openvla-7b+${DATA_ROOT}+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug/"

PIDS=()  # Array to hold process IDs
for ((i=0; i<$NUM_GPUS; i++))
do
    START_TASK_ID=$(($i * $TASKS_PER_GPU))
    END_TASK_ID=$(($START_TASK_ID + $TASKS_PER_GPU))
    GPU_ID=$i

    if [ $i -eq $(($NUM_GPUS - 1)) ]; then
        # Ensure the last GPU processes any remaining tasks
        END_TASK_ID=$TOTAL_TASKS
    fi

    echo "Starting process on GPU $GPU_ID for tasks $START_TASK_ID to $END_TASK_ID"
    python run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint $CKPT_PATH \
        --task_suite_name $DATA_NAME \
        --center_crop True \
        --run_id_note "gpu$GPU_ID" \
        --start_task_id $START_TASK_ID \
        --end_task_id $END_TASK_ID \
        --gpu_id $GPU_ID \
        --use_wandb False \
        > "log_gpu${GPU_ID}.txt" 2>&1 &  # Run in the background
    PIDS+=($!)  # Store the PID of the background process
done

# Wait for all background processes to finish
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "All processes have completed."

# Combine the results automatically
python combine_results.py
