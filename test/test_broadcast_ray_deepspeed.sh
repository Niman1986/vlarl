CUDA_VISIBLE_DEVICES=0,1,2,3 python \
    test/test_broadcast_ray_deepspeed.py \
    --vla_path "MODEL/openvla-7b-finetuned-libero-spatial" \
    --dataset_name libero_spatial_no_noops \
    --task_suite_name libero_spatial \
    --num_trials_per_task 1 \
    --run_root_dir "checkpoints/debug/root" \
    --adapter_tmp_dir "checkpoints/debug/adapter" \
    --per_device_train_batch_size 1 \
    --local_rollout_forward_batch_size 1 \
    --local_mini_batch_size 1 \
    --local_rollout_batch_size 1 \
    --actor_num_gpus_per_node "[3]" \
    --num_steps 520 \
    --env_gpu_id 3 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enforce_eager True \
    --enable_prefix_caching False \
    --use_lora True \
    --enable_gradient_checkpointing True \
    --offload True \
    --adam_offload True \
    --deepspeed_stage 2 \
    --save_freq 100 \
    --save_video True