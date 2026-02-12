"""Off-policy buffer."""
import numpy as np
import torch
from harl.common.buffers.off_policy_buffer_base import OffPolicyBufferBase

##ep和fp都是经验回放缓冲区的实现都继承OffPolicyBufferBase基类；主要区别在于处理全局状态的方式
##OffPolicyBufferEP（Environment-Provided）: 这个类使用环境提供的全局状态。在这个类中，全局观察空间（share_obs）和下一个全局观察空间（next_share_obs）的形状是(self.buffer_size, *self.share_obs_shape)。这意味着每个时间步的全局状态都是由环境直接提供的。
##OffPolicyBufferFP（Feature-Pruned）: 这个类使用特征修剪的全局状态。在这个类中，全局观察空间（share_obs）和下一个全局观察空间（next_share_obs）的形状是(self.buffer_size, self.num_agents, *self.share_obs_shape)。这意味着每个时间步的全局状态是由每个智能体的局部观察组合而成的，每个智能体都有自己的全局状态视图。
##总的来说二者区别在于如何处理和存储全局状态；
##OffPolicyBufferEP区别在于如何处理和存储全局状态，使用环境直接提供的全局状态，使用每个智能体的局部观测而成的全局状态
##经验回放缓冲区实现，继承自基类，主要区别在于处理和存储全局状态
#特征修剪全局状态，类中全局状态全局观察空间和下一个全局观察空间
class OffPolicyBufferEP(OffPolicyBufferBase):
    """Off-policy buffer that uses Environment-Provided (EP) state."""

    def __init__(self, args, share_obs_space, num_agents, obs_spaces, act_spaces):
        """Initialize off-policy buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
            num_agents: (int) number of agents
            obs_spaces: (gym.Space or list) observation spaces
            act_spaces: (gym.Space) action spaces
        """
        super(OffPolicyBufferEP, self).__init__(
            args, share_obs_space, num_agents, obs_spaces, act_spaces
        ) #DSA_Markovenv.py

        # Buffer for share observations
        self.share_obs = np.zeros(
            (self.buffer_size, *self.share_obs_shape), dtype=np.float32
        )

        # Buffer for next share observations
        self.next_share_obs = np.zeros(
            (self.buffer_size, *self.share_obs_shape), dtype=np.float32
        )
        # Buffer for rewards received by agents at each timestep
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.success_hist_1s = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.fail_collision_hist_1s = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.fail_PU_hist_1s = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.fail_TDMA_hist_1s = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.fail_ALOHA_hist_1s = np.zeros((self.buffer_size, 1), dtype=np.float32)
        ##在于如何处理和存储全局状态，前者使用环境直接提供的全局状态，后者使用每个智能体局部观察而成的全局状态
        ##全局观察空间和下一个全局观察空间，全局状态由环境直接提供的
        # Buffer for done and termination flags
        self.dones = np.full((self.buffer_size, 1), False)
        self.terms = np.full((self.buffer_size, 1), False)
    #buffer size确实是经验回放缓冲区的容量上限。

    def sample(self):
        """Sample data for training.
        核心改进：
        1. 替换随机采样为「信道场景感知的优先级采样」，聚焦高价值经验
        2. 保留10%随机探索比例，避免过拟合
        3. 过滤PU占用无效经验，提升样本效率
        Returns:
            同原返回值，仅采样逻辑优化
        """
        self.update_end_flag()  # update the current end flag

        # ========== 前置定义：补全缺失的核心变量 ==========
        # 1. 有效索引范围：仅在当前已存储的经验中采样（原逻辑的valid_indices）
        max_index = self.cur_size - 1  # 新增：定义最大有效索引（解决越界核心）
        valid_indices = np.arange(self.cur_size)
        # 2. 优先级数组（示例：若未存储优先级，默认用均匀优先级，也可替换为TD误差）
        # 注：若要真实优先级，需在缓冲区中新增self.priorities数组，存储每条经验的TD误差
        if not hasattr(self, 'priorities'):
            # 兜底：无优先级时，所有经验权重相等（等价于随机采样）
            self.priorities = np.ones(self.cur_size, dtype=np.float32)
        # 新增：确保self.priorities长度和cur_size一致（避免长度不匹配）
        if len(self.priorities) != self.cur_size:
            self.priorities = np.ones(self.cur_size, dtype=np.float32)

        # ========== 改进1：过滤无效经验（PU占用是环境固有状态，智能体无法学习改变） ==========
        # 筛选出非PU占用的经验索引，减少无效样本干扰
        valid_indices = np.where(self.fail_PU_hist_1s[:self.cur_size].squeeze() != 1)[0]
        # 新增：过滤掉超出max_index的索引（第一道边界防护）
        valid_indices = valid_indices[valid_indices <= max_index]
        # 兜底：若所有经验都是PU占用，保留全部有效索引
        if len(valid_indices) == 0:
            valid_indices = np.arange(self.cur_size)
            valid_indices = valid_indices[valid_indices <= max_index]  # 同步过滤边界

        # ========== 改进2：计算优先级权重（适配信道选择场景的经验价值） ==========
        # 提取信道核心指标（仅取有效经验范围）
        success = self.success_hist_1s[valid_indices].squeeze()  # 信道成功占用（高价值）
        collision = self.fail_collision_hist_1s[valid_indices].squeeze()  # 多智能体冲突（低价值）
        pu = self.fail_PU_hist_1s[valid_indices].squeeze()  # PU占用（无效价值）
        tdma = self.fail_TDMA_hist_1s[valid_indices].squeeze()  # TDMA失败（辅助低价值）
        aloha = self.fail_ALOHA_hist_1s[valid_indices].squeeze()  # ALOHA失败（辅助低价值）

        # 权重公式设计（可在论文中调参对比）：
        # - 成功经验权重+2.0：强化有效策略学习
        # - 冲突/TDMA/ALOHA失败权重-1.5/-1.0/-1.0：弱化无效探索
        # - PU占用权重-1.5：进一步弱化环境固有无效经验
        # - 兜底权重0.1：避免权重为0导致无法采样
        priorities = 1.0 + 2.0 * success - 1.5 * collision - 1.5 * pu - 1.0 * tdma - 1.0 * aloha
        priorities = np.clip(priorities, 0.1, 10.0)  # 限制权重范围，避免极端值

        # ========== 改进3：混合采样（90%优先级+10%随机） ==========
        explore_ratio = 0.1  # 10%随机探索
        n_priority = int(self.batch_size * (1 - explore_ratio))
        n_random = self.batch_size - n_priority

        # 确保采样数量合法（避免n_priority/n_random为0或超过有效长度）
        valid_len = len(valid_indices)
        n_priority = min(n_priority, valid_len)
        n_random = min(n_random, valid_len - n_priority)  # 避免重复采样
        # 兜底：若有效经验不足，直接全量随机采样
        if n_priority + n_random < self.batch_size:
            # 新增：限制采样范围在有效索引内，replace=True允许重复但不越界
            indice = np.random.choice(valid_indices, size=self.batch_size, replace=True)
        else:
            # 1. 优先级采样（90%）：基于优先级概率采样
            # 修复：使用计算好的priorities（而非self.priorities），避免引用错误
            probs = priorities / np.sum(priorities)  # 关键修复：原错误用了self.priorities[valid_indices]
            idx_priority = np.random.choice(
                valid_indices, size=n_priority, replace=False, p=probs
            )

            # 2. 随机采样（10%）：从剩余有效索引中随机选（避免和优先级采样重复）
            remaining_indices = valid_indices[~np.isin(valid_indices, idx_priority)]
            # 新增：确保剩余索引非空
            if len(remaining_indices) == 0:
                remaining_indices = valid_indices
            idx_random = np.random.choice(
                remaining_indices, size=n_random, replace=False
            )

            # 3. 合并并确保长度严格等于batch_size
            indice = np.concatenate([idx_priority, idx_random])
            # 最终兜底：若仍不足，允许重复采样（避免报错）
            if len(indice) < self.batch_size:
                supplement_idx = np.random.choice(
                    valid_indices, size=self.batch_size - len(indice), replace=True
                )
                indice = np.concatenate([indice, supplement_idx])
            # 去重+截断（确保长度精准）
            indice = np.unique(indice)[:self.batch_size]

        # 新增：最后一道边界校验（核心！解决索引越界的最终保障）
        indice = indice[indice <= max_index]
        # 若过滤后长度不足，补充采样（仅从有效索引中）
        if len(indice) < self.batch_size:
            supplement = np.random.choice(
                valid_indices[valid_indices <= max_index],
                size=self.batch_size - len(indice),
                replace=True
            )
            indice = np.concatenate([indice, supplement])[:self.batch_size]

        # ========== 原有逻辑（无修改） ==========
        # get data at the beginning indice
        sp_share_obs = self.share_obs[indice]
        sp_obs = np.array(
            [self.obs[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        sp_actions = np.array(
            [self.actions[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        sp_valid_transitions = np.array(
            [
                self.valid_transitions[agent_id][indice]
                for agent_id in range(self.num_agents)
            ]
        )
        sp_available_actions = np.array(
            [
                self.available_actions[agent_id][indice]
                for agent_id in range(self.num_agents)
            ]
        )
        # compute the indices along n steps
        indices = [indice]
        for _ in range(self.n_step - 1):
            indices.append(self.next(indices[-1]))

        # 新增：对后续n-step索引也做边界校验（防止next函数生成越界索引）
        indices = [idx[idx <= max_index] for idx in indices]
        # 兜底：若某一步索引为空，用前一步索引填充
        for i in range(1, len(indices)):
            if len(indices[i]) == 0:
                indices[i] = indices[i - 1]

        # get data at the last indice
        sp_done = self.dones[indices[-1]]
        sp_term = self.terms[indices[-1]]
        sp_next_share_obs = self.next_share_obs[indices[-1]]
        sp_next_obs = np.array(
            [
                self.next_obs[agent_id][indices[-1]]
                for agent_id in range(self.num_agents)
            ]
        )
        sp_next_available_actions = np.array(
            [
                self.next_available_actions[agent_id][indices[-1]]
                for agent_id in range(self.num_agents)
            ]
        )

        # compute accumulated rewards and the corresponding gamma
        gamma_buffer = np.ones(self.n_step + 1)
        for i in range(1, self.n_step + 1):
            gamma_buffer[i] = gamma_buffer[i - 1] * self.gamma
        sp_reward = np.zeros((self.batch_size, 1))
        gammas = np.full(self.batch_size, self.n_step)
        for n in range(self.n_step - 1, -1, -1):
            now = indices[n]
            # 新增：确保now长度匹配batch_size（避免维度不匹配）
            if len(now) < self.batch_size:
                now = np.pad(now, (0, self.batch_size - len(now)), mode='wrap')[:self.batch_size]
            gammas[self.end_flag[now] > 0] = n + 1
            sp_reward[self.end_flag[now] > 0] = 0.0
            sp_reward = self.rewards[now] + self.gamma * sp_reward
        sp_gamma = gamma_buffer[gammas].reshape(self.batch_size, 1)

        return (
            sp_share_obs,
            sp_obs,
            sp_actions,
            sp_available_actions,
            sp_reward,
            sp_done,
            sp_valid_transitions,
            sp_term,
            sp_next_share_obs,
            sp_next_obs,
            sp_next_available_actions,
            sp_gamma,
        )

    def next(self, indices):
        """Get next indices"""
        return (
            indices + (1 - self.end_flag[indices]) * self.n_rollout_threads
        ) % self.buffer_size

    def update_end_flag(self):
        """Update current end flag for computing n-step return.
        End flag is True at the steps which are the end of an episode or the latest but unfinished steps.
        """
        self.unfinished_index = (
            self.idx - np.arange(self.n_rollout_threads) - 1 + self.cur_size
        ) % self.cur_size
        self.end_flag = self.dones.copy().squeeze()  # (batch_size, )
        self.end_flag[self.unfinished_index] = True
