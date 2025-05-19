import numpy as np
import torch
import torch.nn as nn
import gym
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.utils.discrete_util import gumbel_softmax_sample


'''def get_combined_dim(cent_obs_feature_dim, act_spaces):
    """Get the combined dimension of central observation and individual actions."""
    combined_dim = cent_obs_feature_dim
    combined_dim = len(act_spaces) * 26 + combined_dim  # combined_dim是32
    return combined_dim'''

def get_combined_dim(cent_obs_feature_dim, act_spaces):
    """Get the combined dimension of central observation and individual actions."""
    combined_dim = cent_obs_feature_dim

    # 遍历每个代理的动作空间
    for agent_id, agent_action_space in act_spaces.items():
        # 对于每个代理的动作空间，累加 channel 和 power 的维度
        for action_name, action_space in agent_action_space.items():
            if isinstance(action_space, gym.spaces.Discrete):
                combined_dim += action_space.n  # 对离散空间，使用 n 来获取维度
            elif isinstance(action_space, gym.spaces.Box):
                combined_dim += action_space.shape[0]  # 对连续空间，使用 shape[0] 获取维度
            else:
                raise NotImplementedError(f"Unsupported action space type: {type(action_space)}")

    return combined_dim

    # for space in act_spaces:
    #     if space.__class__.__name__ == "Box":
    #         combined_dim += space.shape[0]
    #     elif space.__class__.__name__ == "Discrete":
    #         combined_dim += space.n
    #     else:
    #         action_dims = space.nvec
    #         for action_dim in action_dims:
    #             combined_dim += action_dim


class ContinuousQNet(nn.Module):
    """Q Network for continuous and discrete action space. Outputs the q value given global states and actions.
    Note that the name ContinuousQNet emphasizes its structure that takes observations and actions as input and outputs
    the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space.
    """

    def __init__(self, args, cent_obs_space, act_space, device=torch.device("cpu")):
        super(ContinuousQNet, self).__init__()
        activation_func = args["activation_func"]
        hidden_sizes = args["hidden_sizes"]
        #print("cccccccccccccccccc")
        #print(cent_obs_space)#Box(-inf, inf, (54,), float32)
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        #print(cent_obs_space)
        if len(cent_obs_shape) == 3:
            self.feature_extractor = PlainCNN(
                cent_obs_shape, hidden_sizes[0], activation_func
            )
            cent_obs_feature_dim = hidden_sizes[0]
        else:
            self.feature_extractor = None
            cent_obs_feature_dim = cent_obs_shape[0]
        # self.action_embedding=nn.Embedding(act_spaces.n,hidden_sizes[0])##创建嵌入层，用于将离散类别变量转换为连续的向量表示
        # self.action_embedding=nn.Embedding(3,hidden_sizes[0])##创建嵌入层，用于将离散类别变量转换为连续的向量表示
        # 使用 set 和 union 获取所有动作空间的并集
        # total_action_space = set().union(*act_spaces)
        # # 转换为列表并排序
        # self.total_action_space = sorted(list(total_action_space))
        #total_action_space[]
        #print(act_spaces,"accc")
        # sizes = (
        #         [get_combined_dim(cent_obs_feature_dim, total_action_space)]
        #         + list(hidden_sizes)
        #         + [1]
        # )
        ##定义多层感知机大小：输入层大小，隐藏层大小，输出层大小
        # sizes = (
        #         [cent_obs_feature_dim] + list(hidden_sizes)+ [len(act_space)]
        # )
        # # sizes = (
        # #         [get_combined_dim(cent_obs_feature_dim, total_action_space)]
        # #         + list(hidden_sizes)
        # #         + [19]
        # # )
        # # sizes = (
        # #     [get_combined_dim(cent_obs_feature_dim, act_spaces)]
        # #     + list(hidden_sizes)
        # #     + [1]
        # # )
        # self.mlp = PlainMLP(sizes, activation_func)
        #self.action_embedding = nn.Embedding(act_space[0].n, hidden_sizes[0])  # max_act_space is the size of the action space

        sizes = (
                [get_combined_dim(cent_obs_feature_dim, act_space)]
                + list(hidden_sizes)
                + [1]
        )
    #print(sizes,"sizes")
        self.mlp = PlainMLP(sizes, activation_func)
        self.to(device)
    ##next_share_obs, next_actions)
    # def forward(self, cent_obs, actions):
    #     if self.feature_extractor is not None:
    #         feature = self.feature_extractor(cent_obs)
    #     else:
    #         feature = cent_obs
    #     #actions = self.action_embedding(actions.long())
    #
    #     concat_x = torch.cat([feature, actions], dim=-1)
    #     q_values = self.mlp(concat_x)
    #     return q_values
    # def forward(self, cent_obs, actions):
    #     if self.feature_extractor is not None:##特征提取器用于处理复杂的观察空间比如图像
    #         feature = self.feature_extractor(cent_obs)
    #     else:
    #         feature = cent_obs
    #     #print(actions,cent_obs,"oo")
    #     #actions = actions.view(len(cent_obs[0]), -1)
    #     # print(actions,"ac=")
    #     # print(feature)
    #     # total_action=self.total_action_space
    #     concat_x = torch.cat([feature, actions], dim=-1)
    #
    #     q_values = self.mlp(concat_x)
    #
    #     #print(q_values,"q_values")
    #     return q_values
    ##定义网络前向传播过程
    # def forward(self, cent_obs, actions):
    def forward(self, cent_obs, actions):
        if self.feature_extractor is not None:
            feature = self.feature_extractor(cent_obs)
        else:
            feature = cent_obs
        concat_x = torch.cat([feature, actions], dim=-1)
        #print(concat_x,len(concat_x))
        #print(concat_x,"d")
        q_values = self.mlp(concat_x)

        # device=q_values.device
        # q_values = gumbel_softmax_sample(q_values, temperature=1.0, device=device)
        #print(q_values,"res")
        # q_values = nn.Sequential(
        #     nn.Linear(19+3,128),
        #     nn.ReLU(),
        #     nn.Linear(128,1)
        # )
        #print(q_values.requires_grad,"grad")
        #print(q_values,"q_values")

        return q_values
