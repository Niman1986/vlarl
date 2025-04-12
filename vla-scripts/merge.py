"""
merge.py

Use the same parameters you used in finetune.py
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


@dataclass
class MergeConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"  # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")  # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"  # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")  # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")  # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16  # Fine-tuning batch size
    max_steps: int = 200_000  # Max number of fine-tuning steps
    save_steps: int = 5000  # Interval for checkpoint saving
    learning_rate: float = 5e-4  # Fine-tuning learning rate
    grad_accumulation_steps: int = 1  # Gradient accumulation steps
    image_aug: bool = True  # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000  # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True  # Whether to save only one checkpoint per run and
    #   continually overwrite the latest checkpoint
    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True  # Whether to use LoRA fine-tuning
    lora_rank: int = 32  # Rank of LoRA weight matrix
    lora_dropout: float = 0.0  # Dropout applied to LoRA weights
    use_quantization: bool = False  # Whether to 4-bit quantize VLA for LoRA fine-tuning
    #   => CAUTION: Reduces memory but hurts performance
    merge_model: bool = False  # Whether to merge LoRA weights into model backbone
    load_model: bool = False  # Whether to load model from checkpoint to resume training

    # Tracking Parameters
    wandb_offline: bool = False  # Whether to run W&B offline
    wandb_project: str = "openvla"  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"  # Name of entity to log under
    run_id_note: Optional[str] = None  # Extra note for logging, Weights & Biases

    # fmt: on

    trust_remote_code: bool = True
    tensor_board: bool = False
    tensor_board_path: str = "logs"
    ppo_train: bool = True
    ppo_coef: float = 0.01

    # PPO Arguments
    clip_param: float = 0.2                                         # PPO clip parameter
    ppo_epoch: int = 5                                              # Number of PPO epochs
    num_mini_batch: int = 4                                         # Number of batches for PPO
    value_loss_coef: float = 1.0                                    # Value loss coefficient
    max_grad_norm: float = 0.5                                      # Max norm of gradients
    huber_delta: float = 10.0                                       # Coefficient of Huber loss
    entropy_coef: float = 0.01                                      # Entropy term coefficient
    use_max_grad_norm: bool = True                                  # Whether to use max norm of gradients
    use_clipped_value_loss: bool = True                             # Whether to clip loss value
    use_huber_loss: bool = True                                     # Whether to use Huber loss
    lr: float = 5e-4                                                # Learning rate
    critic_lr: float = 5e-4                                         # Critic learning rate
    opti_eps: float = 1e-5                                          # RMSprop optimizer epsilon
    gradient_cp_steps: int = 2                                      # Number of steps over which the gradients

    # HACK
    just_copy_needed_files: bool = False
    copy_needed_files: bool = True


@draccus.wrap()
def merge(cfg: MergeConfig) -> None:
    # Configure Unique Experiment ID & Log Directory
    # exp_id = (
    #     f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
    #     f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
    #     f"+lr-{cfg.learning_rate}"
    # )
    # if cfg.use_lora:
    #     exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    # if cfg.use_quantization:
    #     exp_id += "+q-4bit"
    # if cfg.run_id_note is not None:
    #     exp_id += f"--{cfg.run_id_note}"
    # if cfg.image_aug:
    #     exp_id += "--image_aug"

    # Start =>> Build Directories
    # run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    run_dir, adapter_dir = cfg.run_root_dir, cfg.adapter_tmp_dir

    # if not cfg.just_copy_needed_files:
    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    base_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
    merged_vla = merged_vla.merge_and_unload()

    # Overwrite latest checkpoint
    merged_vla.save_pretrained(run_dir)

    del merged_vla, base_vla

    # move the needed files to the run directory
    if cfg.copy_needed_files:
        needed_files = [
            # ".gitattributes",
            "added_tokens.json",
            # "config.json",
            # "configuration_prismatic.py",
            # "dataset_statistics.json",
            # "generation_config.json",
            # "model.safetensors.index.json",
            # "modeling_prismatic.py",
            "preprocessor_config.json",
            # "processing_prismatic.py",
            "README.md",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
        ]
        # copy needed files into the run dir
        for file in needed_files:
            file_path = os.path.join(adapter_dir, file)
            os.system(f"cp {file_path} {run_dir}")
            print(f"Copyed {file_path} to {run_dir}")

    print(f"Merged Model Checkpoint at {run_dir}")

    
if __name__ == "__main__":
    merge()
