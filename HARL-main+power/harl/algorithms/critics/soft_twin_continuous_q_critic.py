"""Soft Twin Continuous Q Critic."""
import numpy as np
import torch
import torch.nn.functional as F
from harl.algorithms.critics.twin_continuous_q_critic import TwinContinuousQCritic
from harl.utils.envs_tools import check


class SoftTwinContinuousQCritic(TwinContinuousQCritic):
    """Soft Twin Continuous Q Critic.
    Critic that learns two soft Q-functions. The action space can be continuous and discrete.
    Note that the name SoftTwinContinuousQCritic emphasizes its structure that takes observations and actions as input
    and outputs the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be
    used in discrete action space.
    """

    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
    ):
        """Initialize the critic."""
        super(SoftTwinContinuousQCritic, self).__init__(
            args, share_obs_space, act_space, num_agents, state_type, device
        )

        self.tpdv_a = dict(dtype=torch.int64, device=device)
        self.auto_alpha = args["auto_alpha"]
        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=args["alpha_lr"]
            )
            self.alpha = torch.exp(self.log_alpha.detach())
        else:
            self.alpha = args["alpha"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_huber_loss = args["use_huber_loss"]
        self.huber_delta = args["huber_delta"]

    def update_alpha(self, logp_actions, target_entropy):
        """Auto-tune the temperature parameter alpha."""
        log_prob = (
            torch.sum(torch.cat(logp_actions, dim=-1), dim=-1, keepdim=True)
            .detach()
            .to(**self.tpdv)
            + target_entropy
        )
        alpha_loss = -(self.log_alpha * log_prob).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = torch.exp(self.log_alpha.detach())

    def get_values(self, share_obs, actions):
        """Get the soft Q values for the given observations and actions."""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        return torch.min(
            self.critic(share_obs, actions), self.critic2(share_obs, actions)
        )

    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        valid_transition,
        term,
        next_share_obs,
        next_actions,
        next_logp_actions,
        gamma,
        value_normalizer=None,
    ):
        """Train the critic.
        Args:
            share_obs: EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            actions: (n_agents, batch_size, dim)
            reward: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            done: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            valid_transition: (n_agents, batch_size, 1)
            term: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            next_share_obs: EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            next_actions: (n_agents, batch_size, dim)
            next_logp_actions: (n_agents, batch_size, 1)
            gamma: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        """
        assert share_obs.__class__.__name__ == "ndarray"
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert next_share_obs.__class__.__name__ == "ndarray"
        assert gamma.__class__.__name__ == "ndarray"

        share_obs = check(share_obs).to(**self.tpdv)
        if self.action_type == "Dict":
            # 假设actions是形状为(n_agents, batch_size)的ndarray，需先重组为字典
            actions_dict = {
                "channel": actions[:, :, 0],  # 信道动作索引
                "power": actions[:, :, 1]  # 功率动作索引
            }
            # 转换为one-hot拼接
            channel_onehot = F.one_hot(
                check(actions_dict["channel"]).to(**self.tpdv_a),
                num_classes=26
            )  # shape: (n_agents, batch_size, 26)
            power_onehot = F.one_hot(
                check(actions_dict["power"]).to(**self.tpdv_a),
                num_classes=4
            )  # shape: (n_agents, batch_size, 4)
            # 拼接并处理无效功率动作（信道动作为25时功率无效）
            # mask = (check(actions_dict["channel"]).to(**self.tpdv_a) != 25).unsqueeze(-1).float()
            # power_onehot = power_onehot * mask.float()
            processed_actions = torch.cat([channel_onehot, power_onehot], dim=-1)  # shape: (n_agents, batch_size, 30)
            # processed_actions是已执行的动作
            # 展平以适应Critic输入（假设Critic已适配30维动作输入）
            n_agents, batch_size, act_dim = processed_actions.shape

            if self.state_type == "FP":
                processed_actions = processed_actions.view(-1, act_dim)
            elif self.state_type == "EP":
                processed_actions = (
                    processed_actions
                    .permute(1, 0, 2)  # -> (batch_size, n_agents, act_dim)
                    .reshape(batch_size, -1)  # -> (batch_size, n_agents * act_dim)
                )
            else:
                processed_actions = processed_actions

            actions = processed_actions

        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        valid_transition = check(np.concatenate(valid_transition, axis=0)).to(
            **self.tpdv
        )
        term = check(term).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)

        if self.action_type == "Dict":
            processed_next_actions = []
            for action_dict in next_actions:
                channel = check(action_dict["channel"]).to(**self.tpdv_a)
                power = check(action_dict["power"]).to(**self.tpdv_a)
                # one-hot 编码
                channel_onehot = F.one_hot(channel, num_classes=26)
                power_onehot = F.one_hot(power, num_classes=4)
                # 屏蔽 channel=25 时的 power
                # mask = (channel != 25).unsqueeze(-1).float()
                # power_onehot = power_onehot * mask
                full_action = torch.cat([channel_onehot, power_onehot], dim=-1)
                full_action = full_action.squeeze(1)
                processed_next_actions.append(full_action)
            next_actions = torch.stack(processed_next_actions, dim=0).to(**self.tpdv_a)
            n_agents, batch_size, act_dim = next_actions.shape
            if self.state_type == "EP":
                # Environment‐Provided 模式，share_obs is (batch, share_obs_dim)
                # 先把维度置为 (batch, n_agents, act_dim)，再 flatten
                next_actions = (
                    next_actions
                    .permute(1, 0, 2)  # -> (batch_size, n_agents, act_dim)
                    .reshape(batch_size, -1)  # -> (batch_size, n_agents * act_dim)
                )
            else:
                # 其他模式，若无特殊需求可保持原样
                next_actions = next_actions
            next_actions = next_actions

        if self.state_type == "EP":
            next_logp_actions_list = next_logp_actions
            logp_cat = torch.cat([logp.unsqueeze(-1) for logp in next_logp_actions_list], dim=1)  # → (batch_size, n_agents)
            joint_logp = logp_cat.sum(dim=1, keepdim=True)  # → (batch_size, 1)
            next_logp_actions = joint_logp.squeeze(-1)

        next_q_values1 = self.target_critic(next_share_obs, next_actions)
        next_q_values2 = self.target_critic2(next_share_obs, next_actions)
        next_q_values = torch.min(next_q_values1, next_q_values2)
        if self.use_proper_time_limits:
            if value_normalizer is not None:
                q_targets = reward + gamma * (
                    check(value_normalizer.denormalize(next_q_values)).to(**self.tpdv)
                    - self.alpha * next_logp_actions
                ) * (1 - term)
                value_normalizer.update(q_targets)
                q_targets = check(value_normalizer.normalize(q_targets)).to(**self.tpdv)
            else:
                q_targets = reward + gamma * (
                    next_q_values - self.alpha * next_logp_actions
                ) * (1 - term)
        else:
            if value_normalizer is not None:
                q_targets = reward + gamma * (
                    check(value_normalizer.denormalize(next_q_values)).to(**self.tpdv)
                    - self.alpha * next_logp_actions
                ) * (1 - done)
                value_normalizer.update(q_targets)
                q_targets = check(value_normalizer.normalize(q_targets)).to(**self.tpdv)
            else:
                q_targets = reward + gamma * (
                    next_q_values - self.alpha * next_logp_actions
                ) * (1 - done)
        if self.use_huber_loss:
            if self.state_type == "EP":
                critic_loss1 = torch.mean(
                    F.huber_loss(
                        self.critic(share_obs, actions),
                        q_targets,
                        delta=self.huber_delta,
                    )
                )
                critic_loss2 = torch.mean(
                    F.huber_loss(
                        self.critic2(share_obs, actions),
                        q_targets,
                        delta=self.huber_delta,
                    )
                )
        else:
            if self.state_type == "EP":
                critic_loss1 = torch.mean(
                    F.mse_loss(self.critic(share_obs, actions), q_targets)
                )
                critic_loss2 = torch.mean(
                    F.mse_loss(self.critic2(share_obs, actions), q_targets)
                )
        critic_loss = critic_loss1 + critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {"critic": critic_loss.item()}

