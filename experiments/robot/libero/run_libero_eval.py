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
from libero.libero import benchmark
from termcolor import cprint, colored
import wandb
import pprint
import json
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import imageio
import cv2


# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
current_path = os.getcwd()
print("Workspace:", current_path)

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
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



def add_info_board(img, thought, step_num, action):
    """
    Usage example:
    img = add_info_board(original_image, thought, step_number, action)
    """
    # Create a board on the right
    board_width = 280
    board_height = img.shape[0]
    board = np.ones((board_height, board_width, 3), dtype=np.uint8) * 255   # white
    
    # Combine image and board horizontally
    combined_img = np.hstack((img, board))
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    text_color = (0, 0, 0)  # Black text
    line_height = 15        # Height of each text line
    margin_left = img.shape[1] + 10  # Start text after original image
    character_limit = 40  # Maximum characters per line
    
    # Extract and format thought
    try:
        preprocessed_thought = thought.split("SUBTASK:")[1].split("MOVE:")[0].strip()
    except:
        preprocessed_thought = thought[:character_limit * 4] 
    
    # Split thought into multiple lines if too long
    thought_lines = []
    words = preprocessed_thought.split()
    current_line = []
    
    for word in words:
        current_line.append(word)
        if len(' '.join(current_line)) > character_limit:  # character limit per line
            thought_lines.append(' '.join(current_line[:-1]))
            current_line = [word]
    if current_line:
        thought_lines.append(' '.join(current_line))
    
    # Draw title
    cv2.putText(combined_img, "Current Status", (margin_left, 30), 
                font, font_scale + 0.2, text_color, 2, cv2.LINE_AA)
    
    # Draw step number
    cv2.putText(combined_img, f"Step: {step_num}", (margin_left, 60),
                font, font_scale, text_color, 1, cv2.LINE_AA)
    
    # Draw thought
    y_position = 100
    cv2.putText(combined_img, "Thought:", (margin_left, y_position),
                font, font_scale, text_color, 1, cv2.LINE_AA)   # NOTE: non-ASCII action string may be garbled
    
    for line in thought_lines:
        y_position += line_height
        cv2.putText(combined_img, line, (margin_left, y_position),
                   font, font_scale, text_color, 1, cv2.LINE_AA)
    
    # Draw action
    y_position += line_height * 2
    cv2.putText(combined_img, "Action:", (margin_left, y_position),
                font, font_scale, text_color, 1, cv2.LINE_AA)
    y_position += line_height
    cv2.putText(combined_img, action, (margin_left, y_position),
                font, font_scale*0.7, text_color, 1, cv2.LINE_AA)   # digit size smaller
    
    return combined_img


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
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize Time Recorder
    timer = TimingManager()

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    num_tasks = min(cfg.num_tasks_per_suite, num_tasks_in_suite)

    if num_tasks < num_tasks_in_suite:
        cprint(f"[Warning] Only evaluating {num_tasks} tasks out of {num_tasks_in_suite}", "red")

    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            pre_thought = None
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                # try:
                if True:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,  # [224, 224, 3]
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),  # [8]
                    }

                    # Query model to get action

                    # print(f"model.norm_stats: {model.norm_stats}")

                    with torch.no_grad(), timer.timer("model_inference"):   # ~0.6s on RTX 3090
                        action, thought = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            pre_thought=pre_thought if cfg.return_thought else None,
                            processor=processor,
                        )
                    if cfg.verbose:
                        cprint(f"timestep: {t}", "cyan")
                        if pre_thought is not None:
                            cprint("Thought:", "cyan")
                        else:
                            cprint("Thought Again:", "yellow")
                        print(thought)

                    # Save preprocessed image for replay video
                    if cfg.return_thought:
                        img = add_info_board(img, thought, t - cfg.num_steps_wait, str(action))

                        ### DEBUG
                        # rollout_dir = f"./rollouts/{DATE}"
                        # os.makedirs(rollout_dir, exist_ok=True)
                        # processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
                        # mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={total_episodes}--success={done}--task={processed_task_description}.mp4"
                        # image_dir = mp4_path.replace(".mp4", "")
                        # os.makedirs(image_dir, exist_ok=True)
                        # img_path = f"{image_dir}/{t}.png"
                        # imageio.imwrite(img_path, img)
                        # cprint(f"Saved image to {img_path}", "green")

                    replay_images.append(img)

                    # refresh the subgoal if needed
                    if cfg.return_thought:
                        pre_thought = thought.split("Out:")[1].split("ACTION:")[0].rstrip()  # remove the last action part and eos token </s>, e.g., [:-11]

                        if t % cfg.subgoal_steps == 0 or pre_thought == None:  # generate another subgoal
                            # thought: <s> In: What action ... Out: PLAN: ... ACTION: ...
                            # -> pre_thought: PLAN: ... ACTION:
                            if cfg.verbose:
                                cprint(f"Refresh this subgoal: {pre_thought}", "yellow")
                            pre_thought = None
                        
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    if cfg.verbose:
                        cprint(f"Action: {action}", "cyan")

                    # Execute action in environment
                    with timer.timer("env_step"):
                        obs, reward, done, info = env.step(action.tolist())
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                # except Exception as e:
                #     cprint(f"Caught exception: {e}", "red")
                #     log_file.write(f"Caught exception: {e}\n")
                #     break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            rollout_dir = f"./rollouts/{DATE}"
            os.makedirs(rollout_dir, exist_ok=True)
            processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
            mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={total_episodes}--success={done}--task={processed_task_description}.mp4"

            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, mp4_path=mp4_path,
            )
            if cfg.save_images:
                image_dir = mp4_path.replace(".mp4", "")
                os.makedirs(image_dir, exist_ok=True)
                for i, img in enumerate(replay_images):
                    img_path = f"{image_dir}/{i:03d}.png"
                    imageio.imwrite(img_path, img)

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final task results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

        log_infos = {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
        time_infos = timer.get_log()
        log_infos.update(time_infos)

        if cfg.use_wandb:
            wandb.log(**log_infos)
        else:
            pprint.pprint(log_infos)

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    time_infos = timer.get_log()
    log_infos = {
        "success_rate/total": float(total_successes) / float(total_episodes),
        "num_episodes/total": total_episodes,
    }
    log_infos.update(time_infos)
    if cfg.use_wandb:
        wandb.log(**log_infos)
        wandb.save(local_log_filepath)
    else:
        pprint.pprint(log_infos)

    env.close()
    cprint("Evaluation complete!", "green")


if __name__ == "__main__":
    eval_libero()
