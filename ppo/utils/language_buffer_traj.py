import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler  #, DistributedSampler

from torch.utils.data import DataLoader
from transformers import AutoProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder, QwenPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class LanguageBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    """

    def __init__(self, args, num_agents, pad_token_id):
        self.args = args    # cfg
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        # self._use_popart = args.use_popart
        # self._use_valuenorm = args.use_valuenorm
        # self.algo = args.algorithm_name
        self.num_agents = num_agents    # n_agents is always 1

        self.max_new_tokens = args.max_new_tokens
        self.max_context_length = args.max_context_length
        self.vacab_size = args.vacab_size
        self.pad_token_id = pad_token_id

        # self.obs = np.empty((self.episode_length + 1, self.n_rollout_threads, num_agents), dtype=np.object_)
        # self.pixel_values = np.empty((self.episode_length + 1, self.n_rollout_threads, 224, 224, 3), dtype=np.uint8)
        # self.actions = np.empty((self.episode_length, self.n_rollout_threads, num_agents, 7), dtype=np.float32) # real 7-DoF robot action
        # self.action_tokens = np.empty((self.episode_length, self.n_rollout_threads, num_agents, self.max_new_tokens), dtype=np.int64)
        # self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, num_agents), dtype=np.float32)

        # remove num_agents
        self.obs = np.empty((self.episode_length + 1, self.n_rollout_threads), dtype=np.object_)
        self.pixel_values = np.empty((self.episode_length + 1, self.n_rollout_threads, 224, 224, 3), dtype=np.uint8)
        self.actions = np.empty((self.episode_length, self.n_rollout_threads, 7), dtype=np.float32) # real 7-DoF robot action
        self.action_tokens = np.empty((self.episode_length, self.n_rollout_threads, self.max_new_tokens), dtype=np.int64)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads), dtype=np.float32)
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads), dtype=np.float32)
        
        # # for action-level ppo and grpo
        # self.action_level_v_values = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents), dtype=np.float32)
        # self.action_level_returns = np.zeros((self.episode_length, self.n_rollout_threads, num_agents), dtype=np.float32)
        self.action_level_v_values = np.zeros((self.episode_length + 1, self.n_rollout_threads), dtype=np.float32)
        self.action_level_returns = np.zeros((self.episode_length, self.n_rollout_threads), dtype=np.float32)
        self.action_level_advantages = np.zeros_like(self.action_level_returns)
        self.action_level_log_probs = np.zeros_like(self.action_level_returns)
        
        self.step = 0

        if args.offline_ratio > 0:
            self.demo_iterator = self.load_dataset(args)

    def load_dataset(self, cfg):
        """
        Load offline demonstration dataset for mixed training.
        """
        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
        
        # Create Action Tokenizer with appropriate type
        if 'qwen' not in cfg.vla_path:
            action_tokenizer = ActionTokenizer(processor.tokenizer)
        else:
            action_tokenizer = ACTION_TOKENIZERS["extra_action_tokenizer"](processor.tokenizer)
            print("Using extra action tokenizer for QWEN model")

        # Select appropriate prompt builder
        if 'qwen' in cfg.vla_path:
            prompt_builder_fn = QwenPromptBuilder
        elif 'v01' in cfg.vla_path:
            prompt_builder_fn = VicunaV15ChatPromptBuilder
        else:
            prompt_builder_fn = PurePromptBuilder

        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=prompt_builder_fn,
            print_prompt_limit=0,
            use_cot=cfg.use_cot,
        )
        
        # Calculate appropriate batch size for offline data
        offline_batch_size = max(1, int(cfg.mini_batch_size * cfg.offline_ratio))
        
        vla_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=(224, 224),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
        )

        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, 
            processor.tokenizer.pad_token_id, 
            padding_side="right"
        )
        
        dataloader = DataLoader(
            vla_dataset,
            batch_size=offline_batch_size,  # Use calculated offline batch size
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        )
        
        return iter(dataloader)

    def get_obs(self, step):
        obs = {
            "prompts": self.obs[step],
            "pixel_values": self.pixel_values[step],
        }
        return obs

    def insert_appo(self, obs, actions, value_preds, rewards, masks, action_tokens, action_log_probs):
        """
        Insert data into the buffer.
        """
        self.pixel_values[self.step + 1] = np.array(obs["pixel_values"]).copy()
        self.obs[self.step + 1] = np.array(obs['prompts']).reshape(self.args.n_rollout_threads).copy()

        self.actions[self.step] = actions.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()    # for gae
        self.action_tokens[self.step] = action_tokens.copy()
        self.action_level_v_values[self.step] = value_preds.copy()
        self.action_level_log_probs[self.step] = action_log_probs.copy()

        self.step = (self.step + 1) % self.episode_length
        
    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.obs[0] = self.obs[-1].copy()
        self.pixel_values[0] = self.pixel_values[-1].copy()
        # self.masks[0] = self.masks[-1].copy()
    
    def batch_process_grpo(self):
        self.action_level_advantages = (self.rewards - np.mean(self.rewards)) / (np.std(self.rewards) + 1e-8)   #

    def batch_process_reinforce(self):
        cumulative_rewards = 0
        for step in reversed(range(self.episode_length)):
            cumulative_rewards = self.rewards[step] + self.gamma * cumulative_rewards * self.masks[step + 1]
            self.action_level_returns[step] = cumulative_rewards
            self.action_level_advantages[step] = self.action_level_returns[step].copy()
        pass

    def batch_process_appo(self, next_value):
        self.action_level_v_values[-1] = next_value
        gae = 0
        for step in reversed(range(self.episode_length)):   #
            delta = self.rewards[step] \
                + self.gamma * self.action_level_v_values[step + 1] * self.masks[step + 1] \
                    - self.action_level_v_values[step]
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.action_level_returns[step] = self.action_level_v_values[step] + gae
            self.action_level_advantages[step] = gae

    def appo_sampler(self, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for APPO.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        batch_size = self.n_rollout_threads * self.episode_length   # sample the number of collected transistions

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, f"batch_size: {batch_size} < num_mini_batch: {num_mini_batch}"
            mini_batch_size = batch_size // num_mini_batch

        # rand = torch.randperm(batch_size).numpy()
        # rand = np.arange(batch_size)
        # np.random.shuffle(rand)
        # sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), # Samples elements randomly from a given list of indices, without replacement.
            mini_batch_size,
            drop_last=True
        )

        # keep (num_agent, dim)
        # NOTE: we should shift the obs (prompt) and pixel_values
        pixel_values = self.pixel_values[:-1].reshape(-1, *self.pixel_values.shape[2:]) # (B+1, N, 224, 224, 3) -> (B*N, 224, 224, 3)
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])    # (B+1, N) -> (B*N,)

        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        value_preds = self.action_level_v_values[:-1].reshape(-1, *self.action_level_v_values.shape[2:])
        returns = self.action_level_returns.reshape(-1, *self.action_level_returns.shape[2:])
        advantages = self.action_level_advantages.reshape(-1, *self.action_level_advantages.shape[2:])
        log_prob = self.action_level_log_probs.reshape(-1, *self.action_level_log_probs.shape[2:])
        action_tokens = self.action_tokens.reshape(-1, *self.action_tokens.shape[2:])

        for indices in sampler:
            yield (
                obs[indices],
                pixel_values[indices],
                actions[indices],
                log_prob[indices],
                value_preds[indices],
                returns[indices],
                advantages[indices],
                action_tokens[indices]
            )

    def mixed_sampler(self, mini_batch_size):
        """
        Yields both PPO and supervised training data based on offline_ratio.
        """
        # Calculate number of offline samples per batch
        offline_samples = int(mini_batch_size * self.args.offline_ratio)
        online_samples = mini_batch_size - offline_samples

        # Ensure minimum batch sizes
        online_samples = max(1, online_samples) if online_samples > 0 else 0
        offline_samples = max(1, offline_samples) if offline_samples > 0 else 0

        # Get online samples generator if needed
        if online_samples > 0:
            online_generator = self.appo_sampler(mini_batch_size=online_samples)
        else:
            online_generator = iter(())

        # Get offline samples if needed
        if offline_samples > 0 and hasattr(self, 'demo_iterator'):
            try:
                offline_batch = next(self.demo_iterator)
                # Adjust offline batch size if needed
                if len(offline_batch['input_ids']) > offline_samples:
                    offline_batch = {k: v[:offline_samples] for k, v in offline_batch.items()}
            except StopIteration:
                self.demo_iterator = self.load_dataset(self.args)
                offline_batch = next(self.demo_iterator)
                if len(offline_batch['input_ids']) > offline_samples:
                    offline_batch = {k: v[:offline_samples] for k, v in offline_batch.items()}
        else:
            offline_batch = None

        # Yield samples
        for online_batch in online_generator:
            # First yield online PPO batch
            yield online_batch
            
            # Then yield offline supervised batch if available
            if offline_batch is not None:
                yield offline_batch
