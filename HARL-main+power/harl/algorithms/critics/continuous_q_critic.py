"""Continuous Q Critic."""
from copy import deepcopy

import numpy as np
import torch
from harl.models.value_function_models.continuous_q_net import ContinuousQNet
from harl.utils.envs_tools import check
from harl.utils.models_tools import update_linear_schedule


class ContinuousQCritic:
    """Continuous Q Critic.
    Critic that learns a Q-function. The action space is continuous.
    Note that the name ContinuousQCritic emphasizes its structure that takes observations and actions as input and
    outputs the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space. For now, it only supports continuous action space, but we will enhance its capability to
    include discrete action space in the future.
    """

    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cuda:0"),
    ):
        """Initialize the critic."""
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.critic = ContinuousQNet(args, share_obs_space, act_space, device)
        self.target_critic = deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )
        self.turn_off_grad()

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        ##训练过程中对优化器的学习率进行衰减(step和steps，一个整数，表示当前训练步骤、总的训练步骤)
        ##调用update_linear_schedule
        ##这个方法目的是在训练过程中逐渐降低学习率，这是一种常见的训练策略，可以帮助模型在训练早期快速收敛，训练后期通过降低学习率微调模型，避免在最优解附近震荡
        update_linear_schedule(self.critic_optimizer, step, steps, self.critic_lr)

    def soft_update(self):
        """Soft update the target network."""
        for param_target, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def get_values(self, share_obs, actions):
        """Get the Q values."""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        #actions,values=self.critic(share_obs, actions)
        return self.critic(share_obs, actions)

    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        term,
        next_share_obs,
        next_actions,
        gamma,
    ):
        """Train the critic.
        Args:
            share_obs: (np.ndarray) shape is (batch_size, dim)
            actions: (np.ndarray) shape is (n_agents, batch_size, dim)
            reward: (np.ndarray) shape is (batch_size, 1)
            done: (np.ndarray) shape is (batch_size, 1)
            term: (np.ndarray) shape is (batch_size, 1)
            next_share_obs: (np.ndarray) shape is (batch_size, dim)
            next_actions: (np.ndarray) shape is (n_agents, batch_size, dim)
            gamma: (np.ndarray) shape is (batch_size, 1)
        """
        assert share_obs.__class__.__name__ == "ndarray"
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert next_share_obs.__class__.__name__ == "ndarray"
        assert gamma.__class__.__name__ == "ndarray"
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        device = next_actions[0].device  # 获取 next_actions 中第一个张量的设备
        next_actions = [action.to(device) for action in next_actions]  # 确保所有的张量都在同一设备上
        next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
        #print(next_actions)

        #next_actions=check(next_actions).to(**self.tpdv)
        #next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        #print(next_actions,next_share_obs)
        next_q_values = self.target_critic(next_share_obs, next_actions)
        # print(next_q_values,print(gamma),"gaa")
        # print(len(next_q_values),len(gamma))
        #next_q_values=check(next_q_values).to(**self.tpdv)
       # next_q_values = [next_q_value.to(device) for next_q_value in next_q_values]  # 确保所有的张量都在同一设备上
       # print(next_q_values,"q_values")
        # print(next_q_values,len(next_q_values),"l")
        # next_q_values=torch.argmax(next_q_values,dim=1)
        if self.use_proper_time_limits:
            q_targets = reward + gamma * next_q_values * (1 - term)
        else:
            q_targets = reward + gamma * next_q_values * (1 - done)
        critic_loss = torch.mean(
            torch.nn.functional.mse_loss(self.critic(share_obs, actions), q_targets)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save(self, save_dir):
        """Save the model."""
        torch.save(self.critic.state_dict(), str(save_dir) + "/critic_agent" + ".pt")
        torch.save(
            self.target_critic.state_dict(),
            str(save_dir) + "/target_critic_agent" + ".pt",
        )

    def restore(self, model_dir):
        """Restore the model."""
        critic_state_dict = torch.load(str(model_dir) + "/critic_agent" + ".pt")
        self.critic.load_state_dict(critic_state_dict)
        target_critic_state_dict = torch.load(
            str(model_dir) + "/target_critic_agent" + ".pt"
        )
        self.target_critic.load_state_dict(target_critic_state_dict)

    def turn_on_grad(self):
        """Turn on the gradient for the critic."""
        for param in self.critic.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """Turn off the gradient for the critic."""
        for param in self.critic.parameters():
            param.requires_grad = False
