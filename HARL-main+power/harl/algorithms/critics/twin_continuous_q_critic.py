"""Twin Continuous Q Critic."""
import itertools
from copy import deepcopy
import torch
import os
from harl.models.value_function_models.continuous_q_net import ContinuousQNet
from harl.utils.envs_tools import check
from harl.utils.models_tools import update_linear_schedule


class TwinContinuousQCritic:
    """Twin Continuous Q Critic.
    Critic that learns two Q-functions. The action space is continuous.
    Note that the name TwinContinuousQCritic emphasizes its structure that takes observations and actions as input and
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
        device=torch.device("cpu"),
    ):
        """Initialize the critic."""
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.action_type = act_space[0].__class__.__name__
        #self.action_type = act_space['agent_0'].__class__.__name__
        self.critic = ContinuousQNet(args, share_obs_space, act_space, device)
        self.critic2 = ContinuousQNet(args, share_obs_space, act_space, device)

        # ========== 新增代码：加载A环境的双Critic权重（参数化路径版） ==========
        try:
            # 1. 从args读取Critic的完整权重路径（配置文件里的完整路径）
            critic_weight_path = args["critic_weight_dir"]
            # 检查文件是否存在（提前排错）
            if not os.path.exists(critic_weight_path):
                raise FileNotFoundError(f"Critic权重文件不存在：{critic_weight_path}")
            # 加载Critic权重
            critic_weights = torch.load(critic_weight_path, map_location=device)
            self.critic.load_state_dict(critic_weights)

            # 2. 从args读取Critic2的完整权重路径（同样参数化，避免硬编码）
            critic2_weight_path = args["critic2_weight_dir"]
            if not os.path.exists(critic2_weight_path):
                raise FileNotFoundError(f"Critic2权重文件不存在：{critic2_weight_path}")
            # 加载Critic2权重
            critic2_weights = torch.load(critic2_weight_path, map_location=device)
            self.critic2.load_state_dict(critic2_weights)

            print(f"✅ 成功加载A环境双Critic权重到B环境！")
            print(f"  - Critic1路径：{critic_weight_path}")
            print(f"  - Critic2路径：{critic2_weight_path}")
            print(f"  - 设备：{device}")
        except FileNotFoundError as e:
            print(f"⚠️ 未找到A环境Critic权重文件：{e}，使用随机初始权重训练B环境")
        except RuntimeError as e:
            print(f"❌ 加载Critic权重失败：{e}")
            print("请检查A/B环境的share_obs_space/act_space是否一致")

        # ========== 原代码：深拷贝得到两个目标Critic网络 + 冻结梯度 ==========
        # 注意：此时target_critic/target_critic2会拷贝加载后的权重（而非随机权重）
        self.target_critic = deepcopy(self.critic)
        self.target_critic2 = deepcopy(self.critic2)

        for param in self.target_critic.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
        critic_params = itertools.chain(
            self.critic.parameters(), self.critic2.parameters()
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=self.critic_lr,
        )
        self.turn_off_grad()

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        update_linear_schedule(self.critic_optimizer, step, steps, self.critic_lr)

    def soft_update(self):
        """Soft update the target networks."""
        for param_target, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )
        for param_target, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )
    ##        # critic 计算损失函数
    def get_values(self, share_obs, actions):
        """Get the Q values for the given observations and actions."""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        #print(share_obs, actions)
        #print(len(share_obs),len(actions),"getvalues")
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
        gamma = check(gamma).to(**self.tpdv)
        #print(next_share_obs,"shareee")
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        #print(next_actions,"next")
        next_actions = [action.reshape(-1, 1) for action in next_actions]
        device=next_actions[0].device
        next_actions = [action.to(device) for action in next_actions]  # 确保所有的张量都在同一设备上

        next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
        #next_actions = torch.cat(next_actions).to(**self.tpdv)
        #print(next_actions,len(next_actions),"next")

        next_q_values1 = self.target_critic(next_share_obs, next_actions)
        next_q_values2 = self.target_critic2(next_share_obs, next_actions)
        next_q_values = torch.min(next_q_values1, next_q_values2)
        if self.use_proper_time_limits:
            q_targets = reward + gamma * next_q_values * (1 - term)
        else:
            q_targets = reward + gamma * next_q_values * (1 - done)
        critic_loss1 = torch.mean(
            torch.nn.functional.mse_loss(self.critic(share_obs, actions), q_targets)
        )
        critic_loss2 = torch.mean(
            torch.nn.functional.mse_loss(self.critic2(share_obs, actions), q_targets)
        )
        critic_loss = critic_loss1 + critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save(self, save_dir):
        """Save the model parameters."""
        torch.save(self.critic.state_dict(), str(save_dir) + "/critic_agent" + ".pt")
        torch.save(
            self.target_critic.state_dict(),
            str(save_dir) + "/target_critic_agent" + ".pt",
        )
        torch.save(self.critic2.state_dict(), str(save_dir) + "/critic_agent2" + ".pt")
        torch.save(
            self.target_critic2.state_dict(),
            str(save_dir) + "/target_critic_agent2" + ".pt",
        )

    def restore(self, model_dir):
        """Restore the model parameters."""
        critic_state_dict = torch.load(str(model_dir) + "/critic_agent" + ".pt")
        self.critic.load_state_dict(critic_state_dict)
        target_critic_state_dict = torch.load(
            str(model_dir) + "/target_critic_agent" + ".pt"
        )
        self.target_critic.load_state_dict(target_critic_state_dict)
        critic_state_dict2 = torch.load(str(model_dir) + "/critic_agent2" + ".pt")
        self.critic2.load_state_dict(critic_state_dict2)
        target_critic_state_dict2 = torch.load(
            str(model_dir) + "/target_critic_agent2" + ".pt"
        )
        self.target_critic2.load_state_dict(target_critic_state_dict2)

    def turn_on_grad(self):
        """Turn on the gradient for the critic network."""
        for param in self.critic.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """Turn off the gradient for the critic network."""
        for param in self.critic.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False
