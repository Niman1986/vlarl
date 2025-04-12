"""
visualize.py

Script to analyze and visualize action coverage in RLDS datasets for OpenVLA.

Run with:
    python vla-scripts/visualize.py --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> --dataset_name <DATASET_NAME>
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

import draccus
import torch
from termcolor import cprint
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig, AutoImageProcessor
from tqdm import tqdm

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder, QwenPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer, ACTION_TOKENIZERS
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Calculate point density using KDE or binning
from scipy.stats import gaussian_kde

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class VisualizeConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"  # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")  # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"  # Name of dataset to analyze
    
    # Visualization Parameters
    num_samples: int = 1000  # Number of samples to analyze
    plot_save_dir: Path = Path("action_coverage_plots")  # Where to save visualizations
    
    # Dataset Parameters
    shuffle_buffer_size: int = 10000  # Dataloader shuffle buffer size
    image_aug: bool = False  # Whether to use image augmentations (not needed for visualization)
    use_cot: bool = False    # Whether to use COT model
    reasoning_dataset_path: str = ""
    # fmt: on


@draccus.wrap()
def visualize_action_coverage(cfg: VisualizeConfig) -> None:
    print(f"Analyzing action coverage in dataset `{cfg.dataset_name}`")
    
    # Ensure GPU is Available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor (needed for action tokenizer)
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    # Create Action Tokenizer
    if 'qwen' not in cfg.vla_path:
        action_tokenizer = ActionTokenizer(processor.tokenizer)
    else:
        action_tokenizer: ActionTokenizer = ACTION_TOKENIZERS["extra_action_tokenizer"](processor.tokenizer)
        cprint(f"Using extra action tokenizer for QWEN model", "yellow")

    # Determine prompt builder based on model
    if 'qwen' in cfg.vla_path:
        prompt_builder_fn = QwenPromptBuilder
    elif 'v01' in cfg.vla_path:
        prompt_builder_fn = VicunaV15ChatPromptBuilder
    else:
        prompt_builder_fn = PurePromptBuilder

    # Create batch transform
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=prompt_builder_fn,
        print_prompt_limit=0,
        use_cot=cfg.use_cot,
    )
    
    # Load dataset
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=(224, 224),  # Standard resolution
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        use_cot=cfg.use_cot,
        reasoning_dataset_path=cfg.reasoning_dataset_path,
    )
    
    # Create directory for plots
    # clear the directory
    if os.path.exists(cfg.plot_save_dir):
        shutil.rmtree(cfg.plot_save_dir)
    os.makedirs(cfg.plot_save_dir, exist_ok=True)
    
    # Analyze action coverage
    action_vectors = []
    action_tokens = []
    
    print(f"Collecting action statistics from {cfg.num_samples} samples...")
    for i, sample in enumerate(tqdm(vla_dataset)):
        if i >= cfg.num_samples:
            break
            
        # Get action tokens from sample
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        
        # Find action tokens in labels (those with value > action_token_begin_idx)
        action_mask = labels > action_tokenizer.action_token_begin_idx
        if torch.any(action_mask):
            action_token_ids = labels[action_mask].cpu().numpy()
            action_tokens.append(action_token_ids)
            
            # Decode tokens to continuous action vector
            continuous_action = action_tokenizer.decode_token_ids_to_actions(action_token_ids)
            action_vectors.append(continuous_action)
    
    # Convert to numpy arrays for analysis
    action_vectors = np.array(action_vectors)
    
    # Print basic statistics
    print(f"\nAction Coverage Analysis for {cfg.dataset_name}")
    print(f"Number of samples analyzed: {len(action_vectors)}")
    
    # Calculate action range and distribution
    if len(action_vectors) > 0:
        action_dims = action_vectors.shape[1]
        print(f"Action dimensions: {action_dims}")

        # Create density-based 2D visualization (XY only)
        plt.figure(figsize=(14, 12))
        
        # Extract just the X and Y dimensions
        xy_data = action_vectors[:, :2]
        
        # Create a styled 2D density visualization
        # First create a colorful background density heatmap
        sns.kdeplot(
            x=xy_data[:, 0], 
            y=xy_data[:, 1],
            cmap="viridis",
            fill=True,
            alpha=0.7,
            levels=15,
            thresh=0.05
        )
        
        # Add contour lines for better visual depth
        contour = sns.kdeplot(
            x=xy_data[:, 0], 
            y=xy_data[:, 1],
            cmap="Oranges_r",
            linewidths=1.5,
            levels=10,
            alpha=0.8
        )
        
        # Calculate the point density for scatter points
        kde = gaussian_kde(xy_data.T)
        density = kde(xy_data.T)
        density = (density - density.min()) / (density.max() - density.min())
        
        # Add scatter points with density-based coloring
        scatter = plt.scatter(
            xy_data[:, 0], 
            xy_data[:, 1],
            c=density,
            cmap='plasma',
            s=70 * density + 15,  # Vary point size by density
            alpha=0.6,
            edgecolor='white',
            linewidth=0.4
        )
        
        # Add a color bar for the scatter points
        cbar = plt.colorbar(scatter, pad=0.02)
        cbar.set_label('Point Density', rotation=270, labelpad=20, fontsize=14)
        
        # Enhance plot appearance
        plt.title(f"XY Action Distribution - {cfg.dataset_name}", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("X Dimension", fontsize=14, labelpad=10)
        plt.ylabel("Y Dimension", fontsize=14, labelpad=10)
        
        # Add grid for reference
        plt.grid(alpha=0.3, linestyle='--')
        
        # Style the plot with a dark background for more visual pop
        plt.gca().set_facecolor('#202030')
        plt.gcf().set_facecolor('#202030')
        
        # Style the tick labels
        plt.tick_params(colors='white', which='both')
        plt.gca().xaxis.label.set_color('white')
        plt.gca().yaxis.label.set_color('white')
        plt.gca().title.set_color('white')
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        
        # Add a subtle outer glow effect by adding a larger, blurred scatter underneath
        if len(xy_data) > 0:
            indices = np.random.choice(len(xy_data), size=min(500, len(xy_data)), replace=False)
            plt.scatter(
                xy_data[indices, 0], 
                xy_data[indices, 1],
                s=200,
                alpha=0.15,
                color='cyan'
            )
        
        plt.tight_layout()
        plt.savefig(f"{cfg.plot_save_dir}/{cfg.dataset_name}_xy_distribution.png", dpi=300, facecolor='#202030')
        
        plt.close()

    else:
        print("No valid action vectors found in the samples.")
    
    print(f"\nAction coverage visualizations saved to {cfg.plot_save_dir}/")


if __name__ == "__main__":
    visualize_action_coverage()