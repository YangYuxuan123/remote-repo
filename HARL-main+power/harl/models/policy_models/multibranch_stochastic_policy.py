import torch
import torch.nn as nn
import gym
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from torch.distributions import Categorical
from harl.models.base.act import ACTLayer

class MultiBranchStochasticPolicy(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.action_space = action_space
        # 共享特征提取层（原MLPBase）
        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)  # 特征提取网络，把原始观测数据转成一个 特征向量

        # 独立动作分支（替换原ACTLayer）
        assert isinstance(action_space, gym.spaces.Dict),  "Action space must be Dict!"
        '''self.channel_branch = nn.Linear(self.hidden_sizes[-1], action_space["channel"].n)
        self.power_branch = nn.Linear(self.hidden_sizes[-1], action_space["power"].n)'''

        hidden = self.hidden_sizes[-1]
        # 信道分支：两层 128 宽度
        self.channel_branch = ACTLayer(
            action_space=self.action_space["channel"],  # 只给它“信道”这一部分
            inputs_dim=self.hidden_sizes[-1],  # 相当于原来的 self.hidden_sizes[-1]
            initialization_method=self.initialization_method,
            gain=self.gain,
            args=self.args,
        )

        # 功率分支：一层 32 宽度
        self.power_branch = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(32),
            nn.Linear(32, action_space["power"].n)
        )

        # 初始化参数
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        """按原逻辑初始化分支权重"""
        for branch in [self.channel_branch, self.power_branch]:
            if self.args["initialization_method"] == "orthogonal":
                nn.init.orthogonal_(branch.weight, gain=self.args["gain"])
                nn.init.constant_(branch.bias, 0)
            elif self.args["initialization_method"] == "xavier":
                nn.init.xavier_uniform_(branch.weight, gain=self.args["gain"])
                nn.init.constant_(branch.bias, 0)

    def forward(self, obs, available_actions=None, stochastic=True):
        """返回复合动作（Dict形式）"""
        obs = check(obs).to(**self.tpdv)
        deterministic = not stochastic
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)  # 共享特征
        channel_action, channel_logp = self.channel_branch(actor_features, available_actions, deterministic)  # channel_logp信道动作的对数概率
        channel_action = channel_action.squeeze(-1)

        # 功率分支（仅在选信道时有效）
        power_logits = self.power_branch(actor_features)
        if available_actions is not None:
            # 可用功率索引在 available_actions 第 26:30 列
            power_mask = available_actions[:, 26:26 + power_logits.size(-1)].bool()
            power_logits = torch.where(power_mask, power_logits, torch.full_like(power_logits, -1e10))

        no_ch_mask = channel_action == (self.action_space["channel"].n - 1)
        if no_ch_mask.any():
            # 将 logits 全部置零，使后续 argmax=0, sample=0
            power_logits[no_ch_mask] = 0.0
        # 5) 构造分布并选择动作
        power_dist = Categorical(logits=power_logits)
        power_action = power_dist.sample() if stochastic else power_logits.argmax(dim=-1)
        # 6) 计算 log-prob
        power_logp = power_dist.log_prob(power_action).unsqueeze(-1)

        return {"channel": channel_action, "power": power_action}

    def get_logits(self, obs, available_actions=None):
        """返回各分支的logits（Dict形式）"""
        obs = check(obs).to(**self.tpdv)
        actor_features = self.base(obs)
        return {
            "channel": self.channel_branch.get_logits(actor_features, available_actions),
            "power": self.power_branch(actor_features)
        }