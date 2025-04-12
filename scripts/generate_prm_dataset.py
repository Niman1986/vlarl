import argparse
import os
import json
import time
import pickle
from PIL import Image
from tqdm import tqdm
import tensorflow_datasets as tfds
from termcolor import cprint
import numpy as np
import shutil
# import google.generativeai as genai
# from google.api_core.exceptions import ResourceExhausted


def get_sharegpt_format(metadata, positive):
    """Convert step data to sharegpt format with binary reward."""
    language_instruction = metadata["language_instruction"]
    image_path = metadata["image_path"]
    step_idx = metadata["step_idx"]
    total_steps = metadata["total_steps"]
    
    # Binary reward based on whether this step is in the final H steps
    binary_reward = "1" if positive else "0"
    
    # Create sharegpt format entry with step information
    messages = [
        {
            "role": "user",
            "content": f"<image>The task is {language_instruction}, is it completed?"
        },
        {
            "role": "assistant",
            "content": binary_reward
        }
    ]
    
    entry = {
        "messages": messages,
        "images": [image_path]
    }
    
    return entry

def build_episode_steps(episode_id, builder, save_root_dir, interval, horizon):
    """Build dataset entries for each step in the episode with given interval."""
    os.makedirs(save_root_dir, exist_ok=True)
    entries = []

    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))
    
    total_steps = len(episode["steps"])
    language_instruction = str(next(iter(episode["steps"]))["language_instruction"].numpy().decode())

    # First pass: Collect all important step indices with their reasons
    important_indices = {}  # Dict to store {step_idx: set of reasons}
    
    # Add steps based on interval
    for idx in range(0, total_steps, interval):
        important_indices[idx] = {"interval"}
    
    # Add final H steps
    for idx in range(max(0, total_steps - horizon), total_steps):
        if idx in important_indices:
            important_indices[idx].add("final_h")
        else:
            important_indices[idx] = {"final_h"}
    
    # Add steps after gripper changes
    prev_gripper_open = None
    delay_after_close = 11  # Steps to wait after gripper closes (open->close)
    delay_after_open = 0    # Steps to wait after gripper opens (close->open)
    window_width_close = 2  # Width of window around target step for close action
    window_width_open = 2   # Width of window around target step for open action
    for step_idx, step in enumerate(episode["steps"]):
        if step_idx > 0:
            # Detect gripper state change
            current_gripper = step["action"][-1]
            if prev_gripper_open is not None and prev_gripper_open != current_gripper:
                # Determine if this is an open->close or close->open transition
                if current_gripper > prev_gripper_open:  # open->close
                    delay = delay_after_close
                    window_width = window_width_close
                else:  # close->open
                    delay = delay_after_open
                    window_width = window_width_open
                
                # Save the step that occurs 'delay' steps after the change
                target_step = min(step_idx + delay, total_steps - 1)
                
                # Add window of steps around target
                window_start = max(0, target_step - window_width)
                window_end = min(total_steps - 1, target_step + window_width)
                for window_step in range(window_start, window_end + 1):
                    if window_step in important_indices:
                        important_indices[window_step].add("gripper_change")
                    else:
                        important_indices[window_step] = {"gripper_change"}
            
                cprint(f"Gripper change at step {step_idx}, target step: {target_step}, window: {window_start} to {window_end}", "yellow")
        prev_gripper_open = step["action"][-1]

    # Create output directory
    os.makedirs(os.path.join(save_root_dir, f"demo_{episode_id}"), exist_ok=True)

    # Second pass: Process only the important steps
    for step_idx, step in enumerate(episode["steps"]):
        if step_idx not in important_indices:
            continue
            
        # Get reasons why this step is important
        reasons = important_indices[step_idx]
        positive = "gripper_change" in reasons or "final_h" in reasons
        
        # Save image for this step
        image = step["observation"]["image"].numpy()
        image_filename = f"{step_idx:05d}.png"
        full_image_path = os.path.join(save_root_dir, f"demo_{episode_id}", image_filename)
        Image.fromarray(image).save(full_image_path)

        metadata = {
            "episode_id": episode_id,
            "step_idx": step_idx,
            "total_steps": total_steps,
            "language_instruction": language_instruction,
            "image_path": full_image_path
        }
        
        entry = get_sharegpt_format(metadata, positive)
        entries.append(entry)
    
    return entries

def generate_dataset(builder, episode_ids, dataset, save_path, dataset_name, interval, horizon):

    # episode_ids = episode_ids[:1]  # for debugging

    pbar = tqdm(episode_ids)
    for i in pbar:
        pbar.set_description(f"Processing episode_id: {i}")
        try:
            save_root_dir = os.path.join(os.path.dirname(save_path), dataset_name)
            entries = build_episode_steps(i, builder, save_root_dir, interval, horizon)
            dataset.extend(entries)
        except Exception as e:
            cprint(f"Failed to process episode_id: {i}, Error: {str(e)}", "red")
            continue
        
        # Save dataset periodically
        # if len(dataset) % 100 == 0:
        #     save_dataset(dataset, save_path)
        #     cprint(f"Saved {len(dataset)} step entries to dataset", "yellow")
    
    # Final save
    save_dataset(dataset, save_path)
    cprint(f"Dataset generation completed! Total step entries: {len(dataset)}", "green", attrs=["bold"])

def save_dataset(dataset, save_path):
    """Save dataset in sharegpt format."""
    # Save main dataset
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    
    # Save dataset info
    json_file = os.path.basename(save_path)
    dataset_name = json_file.replace(".json", "")
    dataset_info = {
        dataset_name: {
            "file_name": json_file,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
    }
    
    info_path = os.path.join(os.path.dirname(save_path), "dataset_info.json")

    # if dataset_name do not exist, add it to the dataset_info
    if not os.path.exists(info_path):
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)
    else:
        with open(info_path, 'r', encoding='utf-8') as f:
            original_dataset_info = json.load(f)
        if dataset_name not in original_dataset_info:
            original_dataset_info.update(dataset_info)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(original_dataset_info, f, indent=2)
    
    cprint(f"Saved dataset to: {save_path}", "cyan")
    cprint(f"Saved dataset info to: {info_path}", "cyan")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="libero_10_no_noops")
    parser.add_argument("--data_root_dir", type=str, default="./data/modified_libero_rlds")
    parser.add_argument("--save_root_dir", type=str, default="./data/reward_model_dataset/")
    parser.add_argument("--interval", type=int, default=5, 
                        help="Interval between steps to save (e.g., 5 means save every 5th step)")
    parser.add_argument("--horizon", type=int, default=10, 
                        help="Number of final steps to label as positive reward")
    args = parser.parse_args()

    builder = tfds.builder(args.dataset_name, data_dir=args.data_root_dir)
    len_train = builder.info.splits["train"].num_examples
    cprint(f"Total training examples: {len_train}", "blue", attrs=["bold"])
    cprint(f"Using step interval: {args.interval}, reward horizon: {args.horizon}", "blue", attrs=["bold"])


    save_root_dir = args.save_root_dir
    os.makedirs(args.save_root_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_root_dir, 
        # f"{args.dataset_name}_reward_interval{args.interval}_horizon{args.horizon}.json"
        f"{args.dataset_name}.json"
    )

    # recover
    if os.path.exists(save_path):
        # with open(save_path, 'r', encoding='utf-8') as f:
        #     dataset = json.load(f)
        # processed_episodes = set(
        #     int(entry["messages"][0]["content"].split("The task is")[0].split("_")[1])
        #     for entry in dataset
        # )
        # episode_ids = list(set(range(len_train)) - processed_episodes)
        # cprint(f"Continuing from existing dataset with {len(dataset)} entries", "yellow")

        # remove files
        os.remove(save_path)
        cprint(f"Removed existing dataset files: {save_path}", "red")
    if os.path.exists(os.path.join(save_root_dir, args.dataset_name)):
        shutil.rmtree(os.path.join(save_root_dir, args.dataset_name))
        cprint(f"Removed existing dataset files: {os.path.join(save_root_dir, args.dataset_name)}", "red")

    # else:
    dataset = []
    episode_ids = list(range(len_train))
    
    generate_dataset(
        builder, 
        episode_ids=episode_ids, 
        dataset=dataset, 
        save_path=save_path,
        dataset_name=args.dataset_name,
        interval=args.interval,
        horizon=args.horizon
    )