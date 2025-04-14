# Usage:
# bash test/test_broadcast_ray_deepspeed.sh

import os
from collections import deque
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from termcolor import cprint
import draccus
import tqdm
import math
import json
import gc
import time
import random
import shutil
import socket
import subprocess
import threading
import numpy as np
import logging
from queue import Empty, Queue
from typing import Any, Callable, Iterator, List, Literal, Optional, Tuple
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor, 
    AutoConfig, 
    AutoImageProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
    BitsAndBytesConfig,
)
from transformers.processing_utils import ProcessorMixin
from transformers.integrations import HfDeepSpeedConfig
# from ppo.conf.config_traj import RLConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig, VISION_BACKBONE_TO_TIMM_ID
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from PIL import Image
import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray.util.collective as collective
from vllm import SamplingParams
from rich.pretty import pprint
from termcolor import cprint
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    disable_dropout_in_model,
    first_true_indices,
    set_seed_everywhere,
    exact_div,
    forward,
    get_reward,
    truncate_response,
    add_special_token,
    masked_mean,
    masked_whiten,
    remove_padding,
    print_rich_single_line_metrics,
    print_rich_table,
)
from ppo.envs.libero_env import VLAEnv
from ppo.models.critic import CriticVLA
from ppo.models.prm import DummyRM, QwenProcessRM
from ppo.utils.util import TimingManager
from ppo.utils.vllm_utils2 import create_vllm_engines, init_process_group
from ppo.utils.ray_utils import ray_noset_visible_devices, get_physical_gpu_id
from ppo.utils.logging_utils import init_logger
from ppo.envs.base import BaseEnv, EnvOutput
from ppo.utils.fsdp_utils import (
    get_fsdp_wrap_policy, 
    offload_fsdp_grad, 
    init_fn, 
    get_init_weight_context_manager, 
    log_gpu_memory_usage
)
# to debug ray instance in vscode, ref: https://github.com/ray-project/ray/issues/41953
# import debugpy
# debugpy.listen(("localhost", 5678))

logger = init_logger(__name__)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

INVALID_LOGPROB = 1.0


@dataclass 
class Args:
    # Common args
    seed: int = 1
    """Seed of the experiment"""

    # VLA Model args
    vla_path: str = "openvla/openvla-7b"
    """Path to OpenVLA model (on HuggingFace Hub)"""
    load_adapter_checkpoint: Optional[str] = None
    """Path to adapter checkpoint to load"""
    pretrained_checkpoint: str = "openvla/openvla-7b"
    """For data collection"""
    # load_in_8bit: bool = False
    # """(For OpenVLA only) Load with 8-bit quantization"""
    # load_in_4bit: bool = False
    # """(For OpenVLA only) Load with 4-bit quantization"""
    model_family: str = "openvla"
    """Model family"""
    use_fast_tokenizer: bool = False
    """Whether to use fast tokenizer"""
    enable_gradient_checkpointing: bool = False
    """Only save important activations to save memory"""

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    """Path to Open-X dataset directory"""
    dataset_name: str = "droid_wipe"
    """Name of fine-tuning dataset (e.g., `droid_wipe`)"""
    run_root_dir: Path = Path("runs")
    """Path to directory to store logs & checkpoints"""
    adapter_tmp_dir: Path = Path("adapter-tmp")
    """Temporary directory for LoRA weights before fusing"""

    # Environment Parameters
    task_suite_name: str = "libero_spatial"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim"""
    num_tasks_per_suite: int = 10
    """Number of tasks per suite"""
    num_trials_per_task: int = 50
    """Number of rollouts per task"""
    n_rollout_threads: int = 10
    """Number of parallel vec environments"""
    max_env_length: int = 0
    """0 for default libero length"""
    env_gpu_id: int = 0
    """GPU id for the vectorized environments"""
    query_length: int = 64
    """Length of the query"""

    # for debugging
    verbose: bool = False
    save_video: bool = True
    """Save video of evaluation rollouts"""

    # Augmentation
    image_aug: bool = True
    """Whether to train with image augmentations"""
    center_crop: bool = True
    """Center crop (if trained w/ random crop image aug)"""

    # LoRA Arguments
    use_lora: bool = False
    """Whether to use LoRA fine-tuning"""
    lora_rank: int = 32
    """Rank of LoRA weight matrix"""
    lora_dropout: float = 0.0
    """Dropout applied to LoRA weights"""
    use_quantization: bool = False
    """Whether to 4-bit quantize VLA for LoRA fine-tuning
    => CAUTION: Reduces memory but hurts performance"""
    merge_model: bool = False
    """Whether to merge LoRA weights into model backbone"""
    load_model: bool = False
    """Whether to load model from checkpoint to resume training"""
    use_multi_adapters: bool = False

    # optimizer args
    eps: float = 1e-5
    """The epsilon value for the optimizer"""
    learning_rate: float = 5e-5
    """The initial learning rate for AdamW optimizer."""
    value_learning_rate: float = 5e-5
    """The initial learning rate for AdamW optimizer."""
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""
    warmup_ratio: float = 0.0
    """Ratio of warmup steps to total steps (takes precedence over `warm_up_steps`)"""

    # various batch sizes
    gradient_accumulation_steps: Optional[int] = None
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: Optional[int] = 2
    """The forward batch size per device (local_micro_batch_size)"""
    per_device_eval_batch_size: Optional[int] = 1
    """The forward batch size per device for evaluation (local_micro_batch_size)"""
    total_episodes: Optional[int] = 100000
    """The total number of episodes in the dataset"""
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_rollout_batch_size: int = 2
    """The number of rollout episodes per iteration per device"""
    rollout_batch_size: Optional[int] = None
    """The number of rollout episodes per iteration"""
    num_training_steps: Optional[int] = None
    """The number of training_steps to train"""
    num_evals: int = 4
    """The number of evaluations to run throughout training"""
    eval_freq: Optional[int] = None
    """The frequency of evaluation steps"""
    local_dataloader_batch_size: Optional[int] = None
    """The batch size per GPU for the dataloader"""
    save_freq: int = -1
    """How many train steps to save the model"""

    # online settings
    num_epochs: int = 1
    """the number of epochs to train (set to 1 to prevent deviating from sft checkpoint)"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: int = 64
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_rollout_forward_batch_size: int = 64
    """per rank no grad forward pass in the rollout phase"""
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""
    reward_model_revision: Optional[str] = None
    """the revision of the reward model"""
    init_value_from_scratch: bool = False
    """whether to initialize the value model from scratch"""

    # generation config
    response_length: int = 8
    """the length of the response"""
    stop_token_id: Optional[int] = None
    """the truncation token id"""
    min_response_length: int = 0
    """stop only after this many tokens"""
    temperature: float = 1.0
    """the sampling temperature"""
    verify_reward_value: float = 10.0
    """the reward value for responses that do not contain `stop_token_id`"""
    penalty_reward_value: float = -1.0
    """the reward value for responses that do not contain `stop_token_id`"""
    non_stop_penalty: bool = False
    """whether to penalize responses that do not contain `stop_token_id`"""
    number_envs_per_task: int = 1
    """the number of samples to generate per prompt, useful for easy-star"""

    # online PPO specific args
    beta: float = 0.0
    """the beta value of the RLHF objective (KL coefficient)"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    cliprange: float = 0.2
    """the clip range"""
    vf_coef: float = 0.1
    """the value function coefficient"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1.0
    """the discount factor (1.0 for sparse rewards, 0.99 for normal case)"""
    lam: float = 1.0
    """the lambda value for GAE (1.0 for sparse rewards, 0.95 for normal case)"""
    kl_estimator: Literal["kl1", "kl2", "kl3"] = "kl1"
    """the KL estimator to use"""
    process_reward_model: bool = False
    """the process reward model (prm), for dense reward"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    norm_adv: bool = False
    """Toggles advantages normalization"""

    # async setting
    async_mode: bool = False
    """Whether to run the generation in async mode which learns from the second latest policy like Cleanba (https://arxiv.org/abs/2310.00036)"""

    # Ray specific
    actor_num_gpus_per_node: List[int] = field(default_factory=lambda: [1])
    """number of gpus per node for actor learner"""
    vllm_num_engines: int = 1
    """number of vLLM Engines, set to 0 to disable vLLM"""
    vllm_tensor_parallel_size: int = 1
    """tensor parallel size of vLLM Engine for multi-GPU inference (1 for single GPU inference)"""
    vllm_enforce_eager: bool = True
    """whether to enforce eager execution for vLLM, set to True to avoid building cuda graph"""
    vllm_sync_backend: str = "nccl"
    """DeepSpeed -> vLLM weight sync backend"""
    enable_prefix_caching: bool = False
    """whether to enable prefix caching"""
    gather_whole_model: bool = True
    """whether to gather the whole model to broadcast (not doable for 70B but can be faster for 8B)"""

    # DeepSpeed-specific args
    deepspeed_stage: int = 0
    """the deepspeed stage"""
    offload: bool = False
    """whether to offload the model to CPU to save GPU memory"""
    adam_offload: bool = False
    """whether to offload the optimizer to CPU to save GPU memory"""

    # wandb and HF tracking configs
    use_wandb: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project"""
    run_id_note: Optional[str] = None
    """Extra note for logging, Weights & Biases"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: Optional[str] = None
    """Where to save the model"""
    checkpoint_output_dir: Optional[str] = None
    """Where to save the model checkpoints in case of preemption"""


def get_num_patches(image_size: int, patch_size: int) -> int:
    grid_length = image_size // patch_size
    return grid_length * grid_length

def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


def get_train_ds_config(
    offload,
    adam_offload=False,
    stage=0,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    disable_trace_cache=True,
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # # ZeRO++
        # "zero_hpz_partition_size": zpg,
        # "zero_quantized_weights": False,
        # "zero_quantized_gradients": False,
    }
    if disable_trace_cache:
        zero_opt_dict["stage3_prefetch_bucket_size"] = 0
        zero_opt_dict["stage3_max_live_parameters"] = 0
        zero_opt_dict["stage3_max_reuse_distance"] = 0

    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {"grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"},
    }

def calculate_runtime_args(args: Args,):
    """calculate (in-place) runtime args such as the effective batch size, word size, etc."""
    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    # args.world_size = accelerator.num_processes
    args.gradient_accumulation_steps = exact_div(
        args.local_mini_batch_size,
        args.per_device_train_batch_size,
        "`local_mini_batch_size` must be a multiple of `per_device_train_batch_size`",
    )
    args.world_size = sum(args.actor_num_gpus_per_node)
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.rollout_batch_size = int(args.local_rollout_batch_size * args.world_size)
    args.n_rollout_threads = args.rollout_batch_size
    args.num_tasks_per_suite = args.rollout_batch_size

    # assert args.num_tasks_per_suite == 10

    args.mini_batch_size = int(args.local_mini_batch_size * args.world_size)
    args.num_training_steps = args.total_episodes // (args.rollout_batch_size)
    args.eval_freq = max(1, args.num_training_steps // args.num_evals)
    # PPO logic: do checks and set up dataloader batch size
    if args.whiten_rewards:
        assert (
            args.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
    args.local_dataloader_batch_size = args.rollout_batch_size

    exp_id = (
        f"{args.vla_path.split('/')[-1]}+{args.dataset_name}"
        f"+b{args.mini_batch_size * args.gradient_accumulation_steps}"
        f"+lr-{args.learning_rate}"
        f"+s-{args.seed}"
    )
    if args.run_id_note is not None:
        exp_id += f"--{args.run_id_note}"
    if args.image_aug:
        exp_id += "--image_aug"
    args.exp_id = exp_id
    cprint(f"Experiment ID: {exp_id}", "green")

    args.unnorm_key = args.task_suite_name

class RayProcess:
    def __init__(self, world_size, rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
        # environment variable for each actor, unless
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set, so
        # set local rank to 0 when the flag is not applicable.
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"

        random.seed(self._rank)
        np.random.seed(self._rank)
        torch.manual_seed(self._rank)

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


@ray.remote(num_gpus=1)
class PolicyTrainerRayProcess(RayProcess):
    def from_pretrained(self, args):
        # Update logger with rank information
        global logger
        logger = init_logger(__name__, self._rank)
        
        # Register OpenVLA model to HF Auto Classes
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        
        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self.args = args

        self._local_rank = int(os.environ["LOCAL_RANK"])
        # logger.info(f"[Actor] set device to local rank: {self._local_rank}")
        # torch.cuda.set_device(self._local_rank)

        logger.info(f"[Actor] RANK={os.environ['RANK']}, WORLD_SIZE={os.environ['WORLD_SIZE']}, "
               f"LOCAL_RANK={os.environ['LOCAL_RANK']}, MASTER_ADDR={os.environ['MASTER_ADDR']}, "
               f"MASTER_PORT={os.environ['MASTER_PORT']}")

        torch.cuda.set_device(self._local_rank)
        deepspeed.init_distributed()

        ds_config = get_train_ds_config(
            offload=args.offload,
            adam_offload=args.adam_offload,
            stage=args.deepspeed_stage,
            bf16=True,
        )
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["train_batch_size"] = args.mini_batch_size
        # Note: dschf is defined in function scope to avoid global effects
        # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        print(f"{dschf=}")

        # Initialize base model
        torch_dtype = torch.bfloat16
        self.model = AutoModelForVision2Seq.from_pretrained(
            args.vla_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False if args.deepspeed_stage == 3 else True,
            trust_remote_code=True,
            # ignore_mismatched_sizes=True,
        )
        # Manually copy gamma parameters to scale_factor after loading
        # from timm.models.vision_transformer import LayerScale
        # for module in self.model.vision_backbone.featurizer.modules():
        #     if isinstance(module, LayerScale) and hasattr(module, 'gamma'):
        #         with torch.no_grad():
        #             module.scale_factor.copy_(module.gamma)
        #         del module.gamma

        # NOTE: 256 is the max image tokens for openvla
        self.hf_config = deepcopy(self.model.config)
        self.max_image_tokens = self.get_max_image_tokens()

        # Load adapter checkpoint if specified
        if args.load_adapter_checkpoint is not None:
            # Load dataset statistics if available
            dataset_statistics_path = os.path.join(args.load_adapter_checkpoint, "dataset_statistics.json")
            if os.path.isfile(dataset_statistics_path):
                with open(dataset_statistics_path, "r") as f:
                    norm_stats = json.load(f)
                self.model.norm_stats = norm_stats
            # Load adapter weights
            self.model = PeftModel.from_pretrained(
                self.model, 
                args.load_adapter_checkpoint,
                is_trainable=True
            )
            logger.info("[Actor] Loaded from adapter checkpoint")
            self.model.print_trainable_parameters()
        # Initialize new LoRA if no checkpoint
        else:
            if args.use_lora:
                lora_config = LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=min(args.lora_rank, 16),
                    lora_dropout=args.lora_dropout,
                    target_modules="all-linear",
                    init_lora_weights="gaussian",
                )
                if args.use_quantization:
                    self.model = prepare_model_for_kbit_training(self.model)
                self.model = get_peft_model(self.model, lora_config)
                
                self.model.print_trainable_parameters()
                logger.info("[Actor] Training from scratch with LoRA")
            else:
                logger.info("[Actor] Training from scratch")

        # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
        self.model.to(torch_dtype)
        disable_dropout_in_model(self.model)
        if args.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        
        log_gpu_memory_usage("[Actor] After model init", rank=self._rank, logger=logger, level=logging.INFO)
        # self.optimizer = torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=args.learning_rate,
        #     eps=args.eps,
        # )
        AdamOptimizer = DeepSpeedCPUAdam if args.adam_offload else FusedAdam
        # AdamOptimizer = torch.optim.AdamW
        optim_params = get_optimizer_grouped_parameters(self.model, weight_decay=0.0)
        self.optimizer = AdamOptimizer(optim_params, lr=args.learning_rate, eps=args.eps)
        num_training_steps = args.num_training_steps * args.num_epochs
        warm_up_steps = args.warm_up_steps
        if args.warmup_ratio >= 0.0:
            warm_up_steps = int(num_training_steps * args.warmup_ratio)
        self.scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=num_training_steps,
        )
        # Verify model contains required action normalization stats
        if args.unnorm_key not in self.model.norm_stats:
            if f"{args.unnorm_key}_no_noops" in self.model.norm_stats:
                args.unnorm_key = f"{args.unnorm_key}_no_noops"
            else:
                raise ValueError(f"Action un-norm key: {args.unnorm_key} not found in VLA `norm_stats`: {self.model.norm_stats.keys()}")
        
        print(f"{ds_config=}")
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            config=ds_config,
            lr_scheduler=self.scheduler,
            dist_init_required=True,
        )
        self.model.train()
        log_gpu_memory_usage("[Actor] After deepspeed wrapping", rank=self._rank, logger=logger, level=logging.INFO)
        # Initialize value model (critic)
        # self.value_model = CriticVLA(args, base_model=self.model)
        # if not args.init_value_from_scratch and args.load_adapter_checkpoint:
        #     critic_weights = os.path.join(args.load_adapter_checkpoint, "critic.pth")
        #     if os.path.exists(critic_weights):
        #         self.value_model.load(critic_weights)
        # disable_dropout_in_model(self.value_model)
        # if args.enable_gradient_checkpointing:
        #     self.value_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        # print("[Value Model] Initialized value model")
        # self.critic_optimizer = torch.optim.AdamW(
        #     self.value_model.parameters(),
        #     lr=args.value_learning_rate,
        #     eps=args.eps,
        # )
        # self.critic_scheduler = get_scheduler(
        #     name=args.lr_scheduler_type,
        #     optimizer=self.critic_optimizer,
        #     num_warmup_steps=warm_up_steps,
        #     num_training_steps=num_training_steps,
        # )
        # self.value_model, self.critic_optimizer, _, self.critic_scheduler = deepspeed.initialize(
        #     model=self.value_model,
        #     optimizer=self.critic_optimizer,
        #     config=ds_config,
        #     lr_scheduler=self.critic_scheduler,
        #     dist_init_required=True,
        # )
        # self.value_model.train()
        self.value_model = None

    def get_max_image_tokens(self) -> int:
        hf_config = self.hf_config
        backbone_id = hf_config.vision_backbone_id
        if backbone_id.startswith("dinosiglip"):
            timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[backbone_id]    # e.g., ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"]
            image_size = hf_config.image_sizes[0]
            patch_size = int(timm_model_ids[0].split("patch")[1].split("_")[0])   # HACK: get patch_size from timm_model_ids
            num_image_tokens = get_num_patches(image_size, patch_size)
        else:
            raise NotImplementedError(f"Unsupported vision backbone: {backbone_id}; only dinosiglip is supported.")
        return num_image_tokens

    def train(
        self,
        vllm_engines: List[ray.actor.ActorHandle],
    ):
        """Main training loop for PPO"""
        logger.info("Starting training loop")
        torch.set_printoptions(precision=4, sci_mode=False)

        args = self.args

        accelerator = Namespace()
        accelerator.process_index = self._rank
        accelerator.num_processes = self._world_size
        accelerator.is_main_process = self._rank == 0
        dist.barrier()

        if accelerator.is_main_process:
            master_address = ray._private.services.get_node_ip_address()
            logger.info(f"Master address: {master_address}")
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            logger.info(f"Master port: {master_port}")
            vllm_num_engines, vllm_tensor_parallel_size = (
                args.vllm_num_engines,
                args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            backend = getattr(args.vllm_sync_backend, "vllm_sync_backend", "nccl")
            group_name = "vllm-inference-group"
            refs = [
                engine.init_process_group.remote(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=i * vllm_tensor_parallel_size + 1,
                    world_size=world_size,
                    group_name=group_name,
                    backend=backend,
                    use_ray=False,
                )
                for i, engine in enumerate(vllm_engines)
            ]
            logger.info(f"[vLLM] Initialized vLLM engines with group name: {group_name}")
            self.model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=group_name,
            )
            ray.get(refs)
            logger.info("[vLLM] Initialized vLLM engines")
        dist.barrier()

        def _broadcast_to_vllm():
            use_prefix_cache = getattr(args, "enable_prefix_caching", False)
            cache_reset_refs = []
            if use_prefix_cache and torch.distributed.get_rank() == 0:
                # clear prefix cache
                for engine in vllm_engines:
                    cache_reset_refs.append(engine.reset_prefix_cache.remote())

            torch.cuda.empty_cache()
            model = self.model.module
            count, num_params = 0, len(list(model.named_parameters()))
            for name, param in model.named_parameters():
                count += 1  # empty_cache at last param
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if args.deepspeed_stage != 3 else param.ds_shape
                    # print(f"🔥🔥🔥 Broadcasting weight: {name} with shape: {shape}")
                    # base_model.model.vision_backbone.featurizer.cls_token
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                        )
                        for engine in vllm_engines
                    ]
                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=args.deepspeed_stage == 3):
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
                        ray.get(refs)

            if cache_reset_refs:
                ray.get(cache_reset_refs)
            torch.cuda.empty_cache()
            torch.distributed.barrier()

        generation_config = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=args.response_length,
            include_stop_str_in_output=False,
            detokenize=False,
            n=1,
            seed=args.seed,
        )
        logger.info("setup async queues")
        response_ids_Q = Queue(maxsize=1)
        param_prompt_Q = Queue(maxsize=1)

        def vllm_generate(
            generation_config: SamplingParams,
            response_ids_Q: Queue,
            param_prompt_Q: Queue,
            num_training_steps: int,
            resume_training_step: int,
        ):
            llm = vllm_engines[0]
            for training_step in range(resume_training_step, num_training_steps + 1):
                g_queries_list = param_prompt_Q.get()
                if g_queries_list is None:
                    break
                prompts = g_queries_list["prompts"]
                prompts = ["<PAD>" + prompt + "▁" for prompt in prompts]
                pixel_values = g_queries_list["pixel_values"]
                generation_start_time = time.time()
                actions, response_ids = ray.get(
                    llm.predict_action.remote(
                        [
                            {
                                "prompt": prompt,
                                "multi_modal_data": {"image": pixel_value},
                            } for prompt, pixel_value in zip(prompts, pixel_values)
                        ],
                        sampling_params=generation_config, 
                        use_tqdm=False,
                        unnorm_key=args.unnorm_key,
                        )
                )
                print(f"🔥🔥🔥 Action generation time: {time.time() - generation_start_time:.2f} seconds")
                response_ids_Q.put((actions, response_ids))

        resume_training_step = 1
        if accelerator.is_main_process:
            thread = threading.Thread(
                target=vllm_generate,
                args=(
                    generation_config,
                    response_ids_Q,
                    param_prompt_Q,
                    args.num_training_steps,
                    resume_training_step,
                ),
            )
            thread.start()
            logger.info("[vLLM] vllm generate thread starts")


        device = torch.device(self._local_rank)
        global_token_obs = {
            "input_ids": torch.zeros(3, 8, device=device, dtype=torch.bfloat16),
            "pixel_values": torch.zeros(3, 6, 224, 224, device=device, dtype=torch.bfloat16),
        }
        if accelerator.is_main_process:
            global_token_obs = {
                "input_ids": torch.randn(3, 8, device=device, dtype=torch.bfloat16),
                "pixel_values": torch.randn(3, 6, 224, 224, device=device, dtype=torch.bfloat16),
            }
        
        logger.info(f"Broadcasting env reset")
        # if torch.distributed.get_rank() == 0:
        for key in ['input_ids', 'pixel_values']:
            # dist.broadcast(global_token_obs[key], src=0, group=self.model.data_parallel_group)
            dist.broadcast(global_token_obs[key], src=0) #group="vllm-inference-group")
            # collective.broadcast(global_token_obs[key], src_rank=0, group_name="vllm-inference-group")
        logger.info(f"Broadcasting env reset done")
        # dist.barrier(group=self.model.data_parallel_group)
        dist.barrier() #group="vllm-inference-group")
        # collective.barrier(group_name="vllm-inference-group")

        rank = dist.get_rank()
        print(f"Tensor on rank {rank} after broadcast: {global_token_obs['input_ids']}")


def kill_ray_cluster_if_a_worker_dies(object_refs: List[Any], stop_event: threading.Event):
    while True:
        if stop_event.is_set():
            break
        for ref in object_refs:
            try:
                ray.get(ref, timeout=0.01)
            except ray.exceptions.GetTimeoutError:
                pass
            except Exception as e:
                print(e)
                print(f"Actor {ref} died")
                time.sleep(120)
                ray.shutdown()
                os._exit(1)  # Force shutdown the process

        time.sleep(30)


class ModelGroup:
    def __init__(
        self,
        pg: PlacementGroup,
        ray_process_cls: RayProcess,
        num_gpus_per_node: List[int],
    ):
        self.pg = pg
        self.ray_process_cls = ray_process_cls
        self.num_gpus_per_node = num_gpus_per_node
        self.num_gpus_per_actor = 1
        self.num_cpus_per_actor = 4
        self.models = []
        world_size = sum(self.num_gpus_per_node)
        master_policy = ray_process_cls.options(
            num_cpus=self.num_cpus_per_actor,
            num_gpus=self.num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=0
            ),
        ).remote(world_size, 0, None, None)

        self.models.append(master_policy)
        master_addr, master_port = ray.get(master_policy.get_master_addr_port.remote())

        def get_bundle_index(rank, num_gpus_per_node):
            """given a rank and a list of num_gpus_per_node, return the index of the bundle that the rank belongs to"""
            bundle_idx = 0
            while rank >= num_gpus_per_node[bundle_idx]:
                rank -= num_gpus_per_node[bundle_idx]
                bundle_idx += 1
            return bundle_idx

        # Setup worker models
        for rank in range(1, world_size):
            print(f"{rank=}, {world_size=}, {rank=}, {master_addr=}, {master_port=}")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.pg,
                placement_group_bundle_index=get_bundle_index(rank, self.num_gpus_per_node),
            )
            worker_policy = ray_process_cls.options(
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
            ).remote(world_size, rank, master_addr, master_port)
            self.models.append(worker_policy)


@draccus.wrap()
def main(args: Args) -> None:
    print(f"PPO Fine-tuning OpenVLA Model `{args.vla_path}` on `{args.dataset_name}`")

    calculate_runtime_args(args)

    # Start =>> Build Directories
    run_dir, adapter_dir = args.run_root_dir / args.exp_id, args.adapter_tmp_dir / args.exp_id
    os.makedirs(run_dir, exist_ok=True)

    args.exp_dir = adapter_dir if args.load_adapter_checkpoint is not None else run_dir
    os.makedirs(args.exp_dir, exist_ok=True)
    video_dir = os.path.join(args.exp_dir, "rollouts")
    cprint(f"Clearing existing videos in {video_dir}", "red")
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith(".mp4"):
                os.remove(os.path.join(video_dir, f))

    set_seed_everywhere(args.seed)

    all_configs = {}
    all_configs.update(**asdict(args))

    # Environment
    envs = VLAEnv(cfg=args, mode="train")

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if args.model_family == "openvla":
        processor = get_processor(args)
    cprint(f"Loaded processor from {args.vla_path}", "green")

    pg = None
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.actor_num_gpus_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    inits = []
    policy_group = ModelGroup(
        pg,
        PolicyTrainerRayProcess,
        args.actor_num_gpus_per_node,
    )
    inits.extend(
        model.from_pretrained.remote(args) for model in policy_group.models
    )
    ray.get(inits)

    max_image_tokens = 256
    max_len = max_image_tokens + args.query_length + args.response_length
    vllm_engines = create_vllm_engines(
        num_engines=args.vllm_num_engines,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        enforce_eager=args.vllm_enforce_eager,
        pretrain=args.vla_path,
        revision=None,
        seed=args.seed,
        enable_prefix_caching=args.enable_prefix_caching,
        max_model_len=max_len,
    )

    print("======== all models initialized =========")
    # ray.get(policy_group.models[0].get_vocab_size.remote())

    refs = []
    for i, policy_model in enumerate(policy_group.models):
        refs.append(
            policy_model.train.remote(
                vllm_engines=vllm_engines,
            )
        )

    # somtimes a worker dies due to CUDA issues, but the rest of the cluster would just hang
    # so we need kill the ray cluster when this happens.
    stop_event = threading.Event()
    threading.Thread(target=kill_ray_cluster_if_a_worker_dies, args=(refs, stop_event)).start()

    ray.get(refs)

    # save model
    ray.shutdown()
    stop_event.set()

    accelerator = Namespace()
    accelerator.is_main_process = True  # hack
    if args.push_to_hub:
        print("Pushing model to hub")
        # TODO: push to hub


if __name__ == "__main__":
    main()
    print("RL Done!")
