"""HAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase


class HAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize HAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(HAPPO, self).__init__(args, obs_space, act_space, device)
        #把args，obs_space，act_space和device参数传递给父类OnPolicyBase。这样，父类的初始化逻辑就会被执行。

        self.clip_param = args["clip_param"] #0.2
        self.ppo_epoch = args["ppo_epoch"] #5
        self.actor_num_mini_batch = args["actor_num_mini_batch"] #1
        self.entropy_coef = args["entropy_coef"] #0.01
        self.use_max_grad_norm = args["use_max_grad_norm"] #True
        self.max_grad_norm = args["max_grad_norm"] #10.0

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()
        # 以上代码描述的是Update Actor Network那个复杂公式

        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()  # add entropy term
        #演员梯度范数可以用来监控和控制训练过程。例如，如果演员梯度范数过大，可能会导致训练不稳定，此时我们可能需要对梯度进行裁剪（即限制其最大值）来防止过大的参数更新。这就是为什么在你的代码中有这样一段：
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        #更新过程中大幅度变换提高学习1稳定性
        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)#
            ##进行优势的标准化处理是强化学习常见处理方式改善学习稳定性和效率，是一个数组表示每个动作优势值  优势值平均值 优势值标准差 零中心化(即减去平均值）
            ##再进行缩放即除以标准差  1e-5为了防止标准差为零导致的除以零错误
            ##优势函数   动作价值函数-值函数  衡量在状态s下执行动作a相比于平均情况下的优势程度 (进行标准化处理可以改善学习稳定性和效率)
        ##使用ppo算法进行训练 策略优化算法 在策略更新时候添加一个限制项防止策略
            #false RNN神经网络

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            # false简单的RNN策略
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            # 前馈神经网络进行训练 是的
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            ##会生成数据生成器 在训练过程中提供数据
            ##循环 迭代 生成对象用于训练
            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    sample
                ) ##策略损失 动作熵 演员梯度范数(actor_grad_norm 神经网络训练概念 用于监控和控制训练过程如果范数过大会导致训练不稳定) 重要性采样权重

                train_info["policy_loss"] += policy_loss.item()  ##获取策略损失标量值
                train_info["dist_entropy"] += dist_entropy.item()##获取动作熵标量值 目标找到最优策略使得累积奖励最大但是训练初期随机性 鼓励策略保持一定随机性防止过早陷入局部最优
                train_info["actor_grad_norm"] += actor_grad_norm #
                train_info["ratio"] += imp_weights.mean()#计算重要性采样平均值

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    #HAPPO是“训练师”，负责：
    #分析历史决策数据（经验回放）
    #计算策略改进方向（损失函数）
    #调整大脑的决策模式（参数更新）
