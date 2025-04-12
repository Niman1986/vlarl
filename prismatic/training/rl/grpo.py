"""
NOT USED
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from prismatic.training.rl.utils import get_gard_norm, huber_loss, mse_loss

# class GRPOTrainer:
#
#     def __init__(self, args, agent):
#         self.tpdv = dict(dtype=torch.float32, device=torch.device("cuda:0"))
#         self.agent = agent
#
#         self.clip_param = args.clip_param
#         self.ppo_epoch = args.ppo_epoch
#         self.num_mini_batch = args.num_mini_batch
#         self.value_loss_coef = args.value_loss_coef
#         self.max_grad_norm = args.max_grad_norm
#         self.huber_delta = args.huber_delta
#         self.entropy_coef = args.entropy_coef
#         self._use_max_grad_norm = args.use_max_grad_norm
#         self._use_clipped_value_loss = args.use_clipped_value_loss
#         self._use_huber_loss = args.use_huber_loss
#         self.lr = args.lr
#         self.critic_lr = args.critic_lr
#         self.opti_eps = args.opti_eps
#         self.gradient_cp_steps = args.gradient_cp_steps
#
#         trainable_params = [param for param in agent.parameters() if param.requires_grad]
#         self.policy_optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, eps=1e-5, weight_decay=0)
#
#     def cal_policy_loss(self, log_prob_infer, log_prob_batch, advantages_batch, entropy):
#
#         log_ratio = log_prob_infer - log_prob_batch
#         imp_weights = torch.exp(log_ratio)
#
#         approx_kl = ((imp_weights - 1) - log_ratio).mean()
#
#         surr1 = -torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
#         surr2 = -imp_weights * advantages_batch
#         surr = torch.max(surr1, surr2)
#         policy_loss = surr.mean() - self.entropy_coef * entropy.mean()
#         return policy_loss, approx_kl
#
#     def ppo_update(self, sample):
#         obs_batch, action_batch, log_prob_batch, \
#             value_preds_batch, return_batch, advantages_batch, action_tokens_batch = sample
#
#         log_prob_batch = torch.from_numpy(log_prob_batch).to("cuda")
#         advantages_batch = torch.from_numpy(advantages_batch).to("cuda")
#         action_tokens_batch = torch.from_numpy(action_tokens_batch).to("cuda")
#         batch_size = obs_batch.shape[0]
#
#         # policy update
#         self.policy_optimizer.zero_grad()
#         cp_batch_size = int(batch_size // self.gradient_cp_steps)
#         total_approx_kl = 0
#         for start in range(0, batch_size, cp_batch_size):
#             end = start + cp_batch_size
#             log_prob_infer, entropy = self.agent.infer_for_action_update(np.concatenate(obs_batch[start:end]),
#                                                                          action_tokens_batch[start:end].view(-1, action_tokens_batch.shape[-1]))
#
#             log_prob_infer = log_prob_infer.view(obs_batch[start:end].shape[0], -1)
#
#             cp_adv_batch = advantages_batch[start:end]
#             cp_adv_batch = (cp_adv_batch - cp_adv_batch.mean()) / (cp_adv_batch.std() + 1e-8)
#
#             entropy = entropy.view(obs_batch[start:end].shape[0], -1)
#             policy_loss, approx_kl = self.cal_policy_loss(log_prob_infer, log_prob_batch[start:end], cp_adv_batch, entropy)
#             total_approx_kl += approx_kl / self.gradient_cp_steps
#
#             # print("policy_loss: ", policy_loss)
#
#             policy_loss /= self.gradient_cp_steps
#             policy_loss.backward()
#         if total_approx_kl > 0.02:
#             self.policy_optimizer.zero_grad()
#             return 0, 0
#
#         policy_grad_norm = nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
#         self.policy_optimizer.step()
#         policy_loss = policy_loss.item()
#         self.policy_optimizer.zero_grad()
#         policy_grad_norm = policy_grad_norm.item()
#
#         return policy_loss, policy_grad_norm
#
#     def train(self, buffer):
#         """
#         Perform a training update using minibatch GD.
#         :param buffer: (SharedReplayBuffer) buffer containing training data.
#         :param update_actor: (bool) whether to update actor network.
#
#         :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
#         """
#         train_info = {}
#         train_info['policy_loss'] = 0
#         train_info['policy_grad_norm'] = 0
#
#         update_time = 0
#         for _ in range(self.ppo_epoch):
#             data_generator = buffer.appo_sampler(self.num_mini_batch)
#             for sample in data_generator:
#                 policy_loss, policy_grad_norm = self.ppo_update(sample)
#                 train_info['policy_loss'] += policy_loss
#                 train_info['policy_grad_norm'] += policy_grad_norm
#                 update_time += 1
#
#         for k in train_info.keys():
#             train_info[k] /= update_time
#
#         return train_info
#
#     def prep_training(self):
#         self.agent.actor().train()
#         self.agent.critic().train()
#
#     def prep_rollout(self):
#         self.agent.actor().eval()
#         self.agent.critic().eval()
#
#     def get_joint_action_log_probs(self, obs, action_tokens, batch_infer=False):
#         pi_logits, _ = self.get_token_logits(obs, action_tokens, batch_infer=batch_infer)
#         pi_log_softmax = torch.log_softmax(pi_logits, dim=-1)
#         action_log_probs = []
#         entropies = []
#         for i in range(pi_logits.shape[0]):
#             act_token_length = self.get_last_token_position(action_tokens[i]) + 1
#             log_softmax_slice = pi_log_softmax[i, :act_token_length, :]
#             action_token_slice = action_tokens[i, :act_token_length]
#             token_log_probs = torch.gather(log_softmax_slice, -1, action_token_slice.unsqueeze(-1)).squeeze(-1)
#             action_log_prob = token_log_probs.sum()
#             action_log_probs.append(action_log_prob)
#
#             entropy = Categorical(logits=pi_logits[i, :act_token_length, :]).entropy().mean()
#             entropies.append(entropy)
#         action_log_probs = torch.stack(action_log_probs)
#         entropies = torch.stack(entropies)
#         return action_log_probs, entropies
#
#     def infer_for_rollout(self, obs, action_preds, action_logits):
#         actions, action_tokens = action_preds, action_logits
#
#         values = np.zeros((obs.shape[0],))  # fake values, grpo does not use critic
#         action_log_probs, _ = self.get_joint_action_log_probs(obs, action_tokens, batch_infer=True)
#         action_tokens = action_tokens.int().cpu().numpy()
#         action_log_probs = action_log_probs.float().cpu().numpy()
#         log_probs = action_log_probs
#
#
#         return actions, action_tokens, values, log_probs


# def ppo_loss(self, old_log_probs, log_probs, rewards, values):
#     # 计算advantages
#     advantages = (rewards - values).unsqueeze(1).expand_as(old_log_probs)
#     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
#
#     # clip loss
#     ratio = torch.exp(log_probs - old_log_probs)
#
#     surr1 = ratio * advantages
#     surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
#     policy_loss = -torch.min(surr1, surr2).mean()
#
#     # 值函数损失
#     value_loss = F.mse_loss(values, rewards)
#
#     # 熵
#     entropy = Categorical(logits=log_probs).entropy().mean()
#
#     # 总损失 PPO损失：策略损失 + 值损失 + 熵奖励
#     total_loss = policy_loss + self.value_loss_coef * value_loss - 0.01 * entropy
#     return total_loss
#
#
# def train_step_chunk(self, thought_logits, action_preds, action_gt, chunk_size=32):
#     batch_size, seq_length, num_actions = thought_logits.size()
#
#     # 存储 actions 和 log_probs
#     actions = torch.zeros(batch_size, seq_length, dtype=torch.long, device=thought_logits.device)
#     log_probs = torch.zeros(batch_size, seq_length, device=thought_logits.device)
#
#     # 采样动作及计算 log_probs
#     for t in range(seq_length):
#         dist = Categorical(logits=thought_logits[:, t, :])
#         sampled_action = dist.sample()
#         actions[:, t].copy_(sampled_action)
#         log_probs[:, t].copy_(dist.log_prob(sampled_action))
#
#     # 计算奖励和价值
#     rewards = self.compute_rewards(action_preds, action_gt).mean(dim=-1)  # [2]
#     values = torch.zeros_like(rewards, device=thought_logits.device)  # [2]
#
#     # 使用 chunk 处理序列损失
#     loss_seq = []
#     for t in range(0, seq_length - chunk_size, chunk_size):
#         # 批量计算 chunk 内损失
#         loss = self.ppo_loss(
#             log_probs[:, t:t + chunk_size],
#             log_probs[:, t + chunk_size:t + chunk_size * 2],
#             rewards,
#             values
#         )
#         loss_seq.append(loss)
#
#     # 计算总损失并反向传播
#     total_loss = torch.mean(torch.stack(loss_seq))
#
#     return total_loss

class ValueNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super(ValueNetwork, self).__init__()

        self.embedding = nn.Linear(vocab_size, hidden_dim)  # [B, V, L] -> [B, V, hidden_dim]

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)  # [B, V, hidden_dim] -> [B, V, hidden_dim]

        self.output_layer = nn.Linear(hidden_dim, 1)  # [B, V, hidden_dim] -> [B, V, 1]

    def forward(self, thought_logits):
        x = self.embedding(thought_logits)  # [B, V, hidden_dim]

        x, _ = self.gru(x)  # [B, V, hidden_dim]

        values = self.output_layer(x).squeeze(-1)  # [B, V, 1] -> [B, V]
        return values


class GRPOTrainer:
    def __init__(self, args, agent, lr=5e-5, gamma=0.99, eps_clip=0.2, value_loss_coef=0.5,
                 entropy_coef=0.01, vocab_size = 32064, chunk_size=32):
        self.agent = agent
        trainable_params = [param for param in agent.parameters() if param.requires_grad]
        self.value_network = ValueNetwork(vocab_size, chunk_size).to(agent.device)
        trainable_params += list(self.value_network.parameters())

        self.policy_optimizer = torch.optim.AdamW(trainable_params, lr=lr, eps=1e-5, weight_decay=0)

        self.gamma = gamma  # 折扣因子
        self.eps_clip = eps_clip  # clip范围
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.chunk_size = chunk_size

    # def compute_rewards(self, preds, ground_truth, mask):
    #     match_reward = ((preds == ground_truth) & mask).float()  # 匹配为1，否则为0
    #     return match_reward

    def compute_returns(self, rewards, batch_size, seq_length, gamma=0.99):
        returns = torch.zeros(batch_size, seq_length, device=rewards.device)
        G = 0  # 初始化最后一个时间步的回报为0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G  # 递推公式
            returns[t] = G
        return returns

    def compute_gae(self, rewards, values, batch_size, seq_length, gamma=0.99, lambda_=0.95):
        advantages = torch.zeros(batch_size, seq_length, device=rewards.device)
        returns = torch.zeros(batch_size, seq_length, device=rewards.device)
        gae = 0

        for t in reversed(range(seq_length)):
            next_value = values[:, t + 1] if t + 1 < seq_length else 0
            delta = rewards[:, t] + gamma * next_value - values[:, t]
            gae = delta + gamma * lambda_ * gae
            advantages[:, t] = gae
            returns[:, t] = advantages[:, t] + values[:, t]  # 返回值和优势函数一致

        return advantages, returns

    def compute_gae_chunk(self, rewards, values, seq_length, chunk_size, gamma=0.99, lambda_=0.95):
        advantages = []
        returns = []
        gae = 0

        for t in reversed(range(0, seq_length, chunk_size)):
            next_value = values[:, t:t + chunk_size]
            delta = rewards[:, t:t + chunk_size] + gamma * next_value - values[:, t:t + chunk_size]
            gae = delta + gamma * lambda_ * gae
            advantages.append(gae)
            returns.append(gae + values[:, t:t + chunk_size])  # 返回值和优势函数一致

        advantages = torch.cat(advantages, dim=-1)
        returns = torch.cat(returns, dim=-1)

        return advantages, returns

    def ppo_loss(self, old_log_probs, log_probs, returns, values, advantages):
        # # 计算advantages
        # advantages = (rewards - values).unsqueeze(1).expand_as(old_log_probs)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # clip loss
        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 值函数损失
        value_loss = F.mse_loss(values, returns)

        # 熵
        entropy = Categorical(logits=log_probs).entropy().mean()

        # 总损失 PPO损失：策略损失 + 值损失 + 熵奖励
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        return total_loss

    def train_step_chunk(self, thought_logits, loss_per_sample):
        batch_size, seq_length, vocab_size = thought_logits.size()

        # # 存储 actions 和 log_probs
        # actions = torch.zeros(batch_size, seq_length, dtype=torch.long, device=thought_logits.device)
        # log_probs = torch.zeros(batch_size, seq_length, device=thought_logits.device)
        # values = torch.zeros(batch_size, seq_length, device=thought_logits.device)
        #
        # # 采样动作及计算 log_probs
        # for t in range(seq_length):
        #     dist = Categorical(logits=thought_logits[:, t, :])
        #     sampled_action = dist.sample()
        #     actions[:, t].copy_(sampled_action)
        #     log_probs[:, t].copy_(dist.log_prob(sampled_action))
        #     value = self.value_network(thought_logits[:, t, :]).squeeze(-1)
        #     values[:, t].copy_(value)


        # 存储 actions 和 log_probs
        dist = Categorical(logits=thought_logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # 计算values
        values = []
        for t in range(0, seq_length, self.chunk_size):
            value = self.value_network(thought_logits[:, t:t + self.chunk_size, :])
            values.append(value)

        values = torch.cat(values, dim=-1)

        # 计算奖励和价值
        rewards = -1. * loss_per_sample # [2]
        rewards = rewards.unsqueeze(-1).expand_as(values)
        # rewards = self.compute_returns(rewards, batch_size, seq_length)

        # values = torch.zeros_like(rewards, device=thought_logits.device)  # [2]

        # 计算 GAE（Generalized Advantage Estimation）
        # advantages, returns = self.compute_gae(rewards, values, batch_size, seq_length)
        advantages, returns = self.compute_gae_chunk(rewards, values, seq_length, self.chunk_size)

        # 使用 chunk 处理序列损失
        loss_seq = []
        for t in range(0, seq_length - self.chunk_size, self.chunk_size):
            loss = self.ppo_loss(
                log_probs[:, t:t + self.chunk_size],
                log_probs[:, t + self.chunk_size:t + self.chunk_size * 2],
                returns[:, t:t + self.chunk_size],
                values[:, t:t + self.chunk_size],
                advantages[:, t:t + self.chunk_size]
            )
            loss_seq.append(loss)

        # 计算总损失并反向传播
        total_loss = torch.mean(torch.stack(loss_seq))

        return total_loss, rewards

    # def ppo_loss(self, old_log_probs, log_probs, rewards, values, advantages):
    #     # [B, 32], [B, 32], [B], [B], [B]
    #     # clip loss
    #     ratio = torch.exp(log_probs - old_log_probs)
    #
    #     # 扩展 advantages 的形状以匹配 ratio
    #     advantages = advantages.unsqueeze(1)  # shape: [batch_size, 1]
    #     advantages = advantages.expand_as(ratio)  # shape: [batch_size, chunk_size]
    #
    #     surr1 = ratio * advantages
    #     surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
    #     policy_loss = -torch.min(surr1, surr2).mean()
    #
    #     # 值函数损失
    #     value_loss = F.mse_loss(values, rewards)
    #
    #     # 熵
    #     entropy = Categorical(logits=log_probs).entropy().mean()
    #
    #     # 总损失 PPO损失：策略损失 + 值损失 + 熵奖励
    #     total_loss = policy_loss + self.value_loss_coef * value_loss - 0.01 * entropy
    #     return total_loss
    #
    # def train_step_chunk(self, thought_logits, action_preds, action_gt, mask, chunk_size=32):
    #     batch_size, seq_length, num_actions = thought_logits.size()
    #
    #     # 存储 actions 和 log_probs
    #     actions = torch.zeros(batch_size, seq_length, dtype=torch.long, device=thought_logits.device)
    #     log_probs = torch.zeros(batch_size, seq_length, device=thought_logits.device)
    #
    #     # 采样动作及计算 log_probs
    #     for t in range(seq_length):
    #         dist = Categorical(logits=thought_logits[:, t, :])
    #         sampled_action = dist.sample()
    #         actions[:, t].copy_(sampled_action)
    #         log_probs[:, t].copy_(dist.log_prob(sampled_action))
    #
    #     # 计算奖励和价值
    #     rewards = self.compute_rewards(action_preds, action_gt, mask).mean(dim=-1)  # [2]
    #     values = torch.zeros_like(rewards, device=thought_logits.device)  # [2]
    #
    #     returns = self.compute_returns(rewards, gamma=self.gamma)
    #     advantages = self.compute_gae(rewards, values, gamma=self.gamma)
    #
    #     # 使用 chunk 处理序列损失
    #     loss_seq = []
    #     for t in range(0, seq_length - chunk_size, chunk_size):
    #          # 批量计算 chunk 内损失
    #         loss = self.ppo_loss(
    #             log_probs[:, t:t + chunk_size],
    #             log_probs[:, t + chunk_size:t + chunk_size * 2],
    #             returns[t:t + chunk_size],
    #             values[t:t + chunk_size],
    #             advantages[t:t + chunk_size]
    #         )
    #         loss_seq.append(loss)
    #
    #     # 计算总损失并反向传播
    #     total_loss = torch.mean(torch.stack(loss_seq))
    #
    #     return total_loss

    def train_step(self, thought_logits, action_preds, action_gt):

        # thought_logits: [2, 256, 32064]
        # 遍历每一个位置并对每个位置的logits进行操作
        batch_size, seq_length, num_actions = thought_logits.size()

        # Initialize storage for log_probs and actions
        actions = torch.zeros(batch_size, seq_length, dtype=torch.long, device=thought_logits.device)
        log_probs = torch.zeros(batch_size, seq_length, device=thought_logits.device)

        for t in range(seq_length):
            # 对于每一个位置，创建分布并采样动作
            dist = Categorical(logits=thought_logits[:, t, :])  # [2, 32064]
            sampled_action = dist.sample()
            actions[:, t].copy_(sampled_action)  # 避免 inplace 赋值问题
            log_prob = dist.log_prob(sampled_action)
            log_probs[:, t].copy_(log_prob)

        # dist: Categorical(probs: torch.Size([2, 32064]), logits: torch.Size([2, 32064]))
        # actions: [2, 256]
        # log_probs: [2, 256]

        # 奖励
        # action_preds [2, 43], action_gt [2, 43]
        rewards = self.compute_rewards(action_preds, action_gt).mean(dim=-1)  # [2] 计算总奖励

        # 价值
        values = torch.zeros_like(rewards, device=thought_logits.device)  # [2]

        loss_seq = []
        # 计算损失并反向传播
        for t in range(seq_length-1):
            loss = self.ppo_loss(log_probs[:,t], log_probs[:,t+1], rewards, values)
            loss_seq.append(loss)

        loss = torch.mean(torch.stack(loss_seq))
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()

    def train_step_once(self, thought_logits, action_preds, action_gt):

        # thought_logits: [2, 256, 32064]
        # dist: Categorical(probs: torch.Size([2, 256, 32064]), logits: torch.Size([2, 256, 32064]))
        dist = Categorical(logits=thought_logits)
        actions = dist.sample()  # [B, L] [2, 256]
        log_probs = dist.log_prob(actions).sum(dim=-1)  # 每个序列的log概率 [2]

        # 计算奖励
        rewards = self.compute_rewards(action_preds, action_gt).sum(dim=-1)

        # 计算值函数
        values = torch.zeros_like(rewards)

        # 获取旧的log_probs和values
        with torch.no_grad():
            old_log_probs = dist.log_prob(actions).sum(dim=-1)
            old_values = values.clone()


        loss = self.ppo_loss(old_log_probs, log_probs, rewards, values, old_values)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()




