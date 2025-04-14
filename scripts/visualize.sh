# !/bin/bash

DATA_ROOT=libero_spatial_no_noops

python vla-scripts/visualize.py \
  --vla_path ./openvla-7b \
  --data_root_dir ./data/modified_libero_rlds \
  --dataset_name ${DATA_ROOT} \
  --num_samples 52970 \
  --plot_save_dir ./action_coverage_plots

  # 52970