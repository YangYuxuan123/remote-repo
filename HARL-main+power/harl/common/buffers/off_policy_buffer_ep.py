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
        self.priorities = np.ones(self.buffer_size, dtype=np.float32)
        self.alpha = 0.6  # priority 强度
        self.beta = 0.4  # IS 修正强度（训练中逐渐增大到1）
        self.epsilon = 1e-6

    def sample(self):
        """
        PER + Importance Sampling 修正版（适用于 HASAC）

        核心改动：
        1. 不删除任何样本（避免分布偏移）
        2. 使用 TD-error 作为优先级
        3. 加入 importance sampling 修正权重
        4. 保持 batch_size 严格一致
        """

        self.update_end_flag()

        # 当前有效范围
        valid_size = self.cur_size
        indices_all = np.arange(valid_size)

        # -------- 1️⃣ 计算采样概率 --------
        priorities = self.priorities[:valid_size]

        # 防止全0
        if priorities.sum() == 0:
            priorities = np.ones_like(priorities)

        # PER公式：p_i^alpha
        scaled_priorities = priorities ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()

        # -------- 2️⃣ 按概率采样 --------
        indice = np.random.choice(
            indices_all,
            size=self.batch_size,
            replace=True,  # PER允许重复
            p=probs
        )

        # -------- 3️⃣ 计算 Importance Sampling 权重 --------
        N = valid_size
        sample_probs = probs[indice]

        weights = (1.0 / (N * sample_probs)) ** self.beta
        weights /= weights.max()  # 归一化，避免梯度爆炸

        # 转 torch tensor 方便外部使用
        # is_weights = torch.FloatTensor(weights).unsqueeze(-1)
        is_weights = weights.reshape(-1, 1).astype(np.float32)

        # ======================================================
        # 下面保持你原始逻辑（完全不动）
        # ======================================================

        sp_share_obs = self.share_obs[indice]
        sp_obs = np.array(
            [self.obs[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        sp_actions = np.array(
            [self.actions[agent_id][indice] for agent_id in range(self.num_agents)]
        )
        sp_valid_transitions = np.array(
            [self.valid_transitions[agent_id][indice]
             for agent_id in range(self.num_agents)]
        )
        sp_available_actions = np.array(
            [self.available_actions[agent_id][indice]
             for agent_id in range(self.num_agents)]
        )

        # -------- n-step ----------
        indices = [indice]
        for _ in range(self.n_step - 1):
            indices.append(self.next(indices[-1]))

        sp_done = self.dones[indices[-1]]
        sp_term = self.terms[indices[-1]]
        sp_next_share_obs = self.next_share_obs[indices[-1]]
        sp_next_obs = np.array(
            [self.next_obs[agent_id][indices[-1]]
             for agent_id in range(self.num_agents)]
        )
        sp_next_available_actions = np.array(
            [self.next_available_actions[agent_id][indices[-1]]
             for agent_id in range(self.num_agents)]
        )

        # -------- 计算 n-step reward ----------
        gamma_buffer = np.ones(self.n_step + 1)
        for i in range(1, self.n_step + 1):
            gamma_buffer[i] = gamma_buffer[i - 1] * self.gamma

        sp_reward = np.zeros((self.batch_size, 1))
        gammas = np.full(self.batch_size, self.n_step)

        for n in range(self.n_step - 1, -1, -1):
            now = indices[n]
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
            is_weights,  # ⭐ 新增：返回IS权重
            indice  # ⭐ 新增：用于更新priority
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
