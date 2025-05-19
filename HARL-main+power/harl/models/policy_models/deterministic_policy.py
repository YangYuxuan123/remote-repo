import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP

np.set_printoptions(threshold=np.inf)
from harl.utils.discrete_util import gumbel_softmax_sample


class DeterministicPolicy(nn.Module):
    """Deterministic policy network for continuous action space."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize DeterministicPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        hidden_sizes = args["hidden_sizes"]
        activation_func = args["activation_func"]
        final_activation_func = args["final_activation_func"]
        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            self.feature_extractor = PlainCNN(
                obs_shape, hidden_sizes[0], activation_func
            )
            feature_dim = hidden_sizes[0]
        else:
            self.feature_extractor = None
            feature_dim = obs_shape[0]
        ##print(action_space,"test")
        # act_dim = action_space.shape[0]
        act_dim = action_space.n##19
        #print(act_dim,"acc")
        #print(act_dim,"act_dim")#19 act_dim

        pi_sizes = [feature_dim] + list(hidden_sizes) + [act_dim]
        #print(pi_sizes,"pi_sizes")#[8, 128, 128, 19] pi_sizes

        self.pi = PlainMLP(pi_sizes, activation_func, final_activation_func)
        # low = torch.tensor(action_space.low).to(**self.tpdv)
        # high = torch.tensor(action_space.high).to(**self.tpdv)
        # self.scale = (high - low) / 2
        # self.mean = (high + low) / 2
        self.to(device)
##基于这个选择动作
    def forward(self, obs):
        # Return output from network scaled to action space limits.
        if self.feature_extractor is not None:
            x = self.feature_extractor(obs)
        else:
            x = obs
        # np.set_printoptions(threshold=np.inf)
        #print(x.cpu().numpy(), "x---")

        x = self.pi(x)
        device=x.device
        x = gumbel_softmax_sample(x, temperature=1.0, device=device)
        # self.to(device)
        return x

# import numpy as np
# import torch
# import torch.nn as nn
# from harl.utils.envs_tools import get_shape_from_obs_space
# from harl.models.base.plain_cnn import PlainCNN
# from harl.models.base.plain_mlp import PlainMLP
#
# np.set_printoptions(threshold=np.inf)
#
#
# ##确定性策略网络用于连续性动作空间
# class DeterministicPolicy(nn.Module):
#     """Deterministic policy network for continuous action space."""
#
#     def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
#         """Initialize DeterministicPolicy model.
#         Args:
#             args: (dict) arguments containing relevant model information.
#             obs_space: (gym.Space) observation space.
#             action_space: (gym.Space) action space.
#             device: (torch.device) specifies the device to run on (cpu/gpu).
#         """
#         super().__init__()
#         self.tpdv = dict(dtype=torch.float32, device=device)#数据类型&设备信息
#         hidden_sizes = args["hidden_sizes"]#
#         activation_func = args["activation_func"]
#         final_activation_func = args["final_activation_func"]
#         #
#         #final_activation_func = args["final_activation_func"]#定义神经网络最后一层激活函数，激活函数用于在神经网络每一层之后应用的非线性函数帮助神经网络学习复杂的模式；这里最后一层使用双曲正切函数作为激活函数，可以将任何实数映射到(-1,1)对于许多需要输出在特定范围内的任务十分有用
#         ##比如神经网络是一个策略网络，用于输出一个连续的动作值，使用双曲正切函数作为最后一层激活函数保证了输出动作值(-1,1)之间
#         obs_shape = get_shape_from_obs_space(obs_space)
#         #根据观察空间的形状设计特征提取器，例如对于观察空间形状是3(对于图像输入高、宽、通道数)，创建CNN对象作为特征提取器
#         if len(obs_shape) == 3:
#             self.feature_extractor = PlainCNN(
#                 obs_shape, hidden_sizes[0], activation_func
#             )
#             feature_dim = hidden_sizes[0]
#         else:
#             self.feature_extractor = None
#             feature_dim = obs_shape[0]
#         self.act_dim = 9#len(action_space[0])#action_space.shape[0]
#         pi_sizes = [feature_dim] + list(hidden_sizes) + [self.act_dim]
#
#         #self.pi = PlainMLP(pi_sizes, activation_func, final_activation_func)
#         #PlainMLP是一个类定义了一个多层感知机模型，全连接神经网络，由多个线性层何非线性激活函数组成
#         #pi_sizes是一个列表定义神经网络每一层大小，列表第一个元素是输入层大小，最后一个元素是输出层大小，中间元素是隐藏层大小
#         #activation_func是一个激活函数，用于每一层输出除了最后一层
#         #nn.Identity是一个函数返回输入值，这里被用作最后一层激活函数，意味着最后一层输出不经过任何激活函数，最后一层是线性的
#         self.pi = PlainMLP(pi_sizes, activation_func, final_activation_func)#创建多层感知机MLP模型，用于计算策略网络的输出
#         ##以下用于计算动作空间的缩放和中心点，在处理连续动作空间时候作用，将神经网络输出缩放到动作空间的范围。
#         # low = torch.tensor(action_space.low).to(**self.tpdv)
#         # high = torch.tensor(action_space.high).to(**self.tpdv)
#         # self.scale = (high - low) / 2
#         # self.mean = (high + low) / 2
#         self.to(device)
#     # def forward(self, obs):
#     #     # Return output from network scaled to action space limits.
#     #     if self.feature_extractor is not None:
#     #         x = self.feature_extractor(obs)
#     #     else:
#     #         x = obs
#     #     x = self.pi(x)
#     #     prob=nn.functional.softmax(x,dim=-1)
#     #     dist=torch.distributions.Categorical(prob)
#     #     #action = torch.argmax(prob, dim=-1)
#     #     action=dist.sample()#策略网络输出的动作概率分布中采样得到的动作
#     #     return action
#     def forward(self, obs):
#         # Return output from network scaled to action space limits.
#         if self.feature_extractor is not None:
#             x = self.feature_extractor(obs)
#         else:
#             x = obs
#         #print(x.cpu().numpy(), "x---")
#         ##q_values
#         x = self.pi(x)
#         #print(x.cpu().numpy(), "x---")
#         # x = torch.max(x, dim=1).values
#         # x = x.unsqueeze(1)
#
#         #print(x.cpu().numpy(), "x---")
#
#         #print(x.cpu().numpy(), "x---")
#
#         #print(x,"test")
#         #x = self.scale * x + self.mean
#         return x
