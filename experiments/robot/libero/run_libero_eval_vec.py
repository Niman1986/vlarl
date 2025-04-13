"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import time
from libero.libero import benchmark
from ppo.envs.libero_env import VLAEnv
from termcolor import cprint, colored
import wandb
import pprint
import json
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import imageio
import cv2

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder, QwenPromptBuilder


# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
current_path = os.getcwd()
print("Workspace:", current_path)

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_actions,
    get_actions_batch,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from ppo.utils.util import TimingManager


@dataclass
class GenerateConfig:
    # fmt: off
    vla_path: str = "openvla-7b"       # OpenVLA model path
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    num_tasks_per_suite: int = 10                    # Number of tasks to evaluate

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    return_thought: bool = False                     # whether return decoded thought chain
    verbose: bool = False                            # Verbose mode for debugging
    subgoal_steps: int = 5                           # Number of steps to take for each subgoal (keep thought chain consistent)
    # fmt: on

    load_adapter_checkpoint: Optional[str] = None    # Path to adapter checkpoint to load
    save_images: bool = False                        # Whether to save images besides videos
    
    n_rollout_threads: int = 10
    max_env_length: int = 0
    save_video: bool = False
    penalty_reward_value: float = 0.0
    non_stop_penalty: bool = False
    verify_reward_value: float = 1.0


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # Load prompt builder
    if 'qwen' in cfg.pretrained_checkpoint:
        prompt_builder_fn = QwenPromptBuilder
        cprint(f"Using QwenPromptBuilder for QWEN model", "yellow")
    elif 'v01' in cfg.pretrained_checkpoint:
        prompt_builder_fn = VicunaV15ChatPromptBuilder
    else:
        prompt_builder_fn = PurePromptBuilder

    if cfg.load_adapter_checkpoint is not None:
        # with open(run_dir / "latest_checkpoint_step.txt", "r") as f:
        #     resume_iteration = int(f.read())
        # start_gradient_step_idx = resume_iteration + 1
        # print(f"Resuming training from iteration {resume_iteration} ...")

        dataset_statistics_path = os.path.join(cfg.load_adapter_checkpoint, "dataset_statistics.json")  # HACK: overwrite dataset_statistics
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                norm_stats = json.load(f)
            model.norm_stats = norm_stats
            cprint(f"Loaded dataset statistics from {dataset_statistics_path}", "green")

        # Load Model Weights
        model = PeftModel.from_pretrained(model, cfg.load_adapter_checkpoint, is_trainable=False)
        model.merge_and_unload()    # accelerate inference
        model.print_trainable_parameters()
        cprint(f"Loaded adapter weights from {cfg.load_adapter_checkpoint}", "green")

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key: {cfg.unnorm_key} not found in VLA `norm_stats`: {model.norm_stats.keys()}! (Please check dataset_statistics.json)"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    # max_pet_name_len = len("openvla-7b")
    # model_pet_name = cfg.pretrained_checkpoint.split('/')[-1][:max_pet_name_len]
    model_pet_name = cfg.load_adapter_checkpoint.split('/')[-1] if cfg.load_adapter_checkpoint else cfg.pretrained_checkpoint.split('/')[-1]
    run_id = f"EVAL-{cfg.task_suite_name}-{model_pet_name}-s-{cfg.seed}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    local_log_dir = os.path.join(cfg.local_log_dir, run_id)
    os.makedirs(local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    cprint(f"Logging to local log file: {local_log_filepath}", "cyan")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # cfg.exp_dir = cfg.pretrained_checkpoint if cfg.load_adapter_checkpoint is None else cfg.load_adapter_checkpoint    # for saved video
    cfg.exp_dir = local_log_dir
    # Clear exp_dir to avoid video duplication
    video_dir = os.path.join(cfg.exp_dir, "rollouts")
    cprint(f"Clearing existing videos in {video_dir}", "red")
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith(".mp4"):
                os.remove(os.path.join(video_dir, f))

    # Initialize vectorized environment
    envs = VLAEnv(cfg, mode="eval")
    timer = TimingManager()

    # Main evaluation loop
    total_episodes = 0
    total_successes = 0
    pre_thought = None
    
    # Reset environments
    obs, _ = envs.reset()
        
    while True:
        # Batch inference
        with torch.no_grad(), timer.timer("model_inference"):
            pixel_values = obs["pixel_values"]
            prompts = obs["prompts"]

            generation_start_time = time.time()
            # actions, _, _ = get_actions(cfg, model, obs=pixel_values, task_label=prompts, pre_thought=None, processor=processor, prompt_builder_fn=prompt_builder_fn)
            actions, _, _ = get_actions_batch(cfg, model, obs=pixel_values, task_label=prompts, pre_thought=None, processor=processor, prompt_builder_fn=prompt_builder_fn)
            print(f"🔥🔥🔥 Action generation time: {time.time() - generation_start_time:.2f} seconds")

        # Step environments
        with timer.timer("env_step"):
            next_obs, rewards, dones, infos = envs.step(actions)
        
        # Update success count
        total_successes += sum([r > 0 for r in rewards])
        
        # Break if all episodes in batch are done
        if np.allclose(envs.initial_state_ids, -1):
        # if next_obs is None:
            break
            
        obs = next_obs
        
    total_episodes = cfg.num_trials_per_task * cfg.num_tasks_per_suite
    
    # Log progress
    success_rate = total_successes / total_episodes
    print(f"Episodes completed: {total_episodes}")
    print(f"Success rate: {success_rate:.2%}")
    log_file.write(f"Episodes completed: {total_episodes}\n")
    log_file.write(f"Success rate: {success_rate:.2%}\n")
    
    # Push total metrics and local log file to wandb
    time_infos = timer.get_log()
    log_infos = {
        "success_rate/total": float(total_successes) / float(total_episodes),
        "num_episodes/total": total_episodes,
    }
    log_infos.update(time_infos)

    for k, v in log_infos.items():
        log_file.write(f"{k}: {v}\n")
    pprint.pprint(log_infos)

    if cfg.use_wandb:
        wandb.log(log_infos)
        wandb.save(local_log_filepath)

    # Cleanup
    log_file.close()
    timer.close()
    envs.close()
    cprint("Evaluation complete!", "green")


if __name__ == "__main__":
    eval_libero()
