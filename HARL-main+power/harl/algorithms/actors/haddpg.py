"""HADDPG algorithm."""
from copy import deepcopy

import numpy as np
import torch
from torch.distributions import Categorical

from harl.models.policy_models.deterministic_policy import DeterministicPolicy
from harl.utils.envs_tools import check
from harl.algorithms.actors.off_policy_base import OffPolicyBase
from harl.utils.discrete_util import gumbel_softmax_sample
from harl.utils.discrete_util import gumbel_softmax
class HADDPG(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        # assert (
        #     act_space.__class__.__name__ == "Box"
        # ), f"only continuous action space is supported by {self.__class__.__name__}."
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.expl_noise = args["expl_noise"]
        self.epsilon=args["epsilon"]
        #self.device=device
        self.actor = DeterministicPolicy(args, obs_space, act_space, device)
        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = True
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        # self.low = torch.tensor(act_space.low).to(**self.tpdv)
        # self.high = torch.tensor(act_space.high).to(**self.tpdv)
        # self.scale = (self.high - self.low) / 2
        # self.mean = (self.high + self.low) / 2
        self.turn_off_grad()
    ##用于获取给定观察值的动作，首先将观测值转换为适合pytorch处理的格式，然后通过策略网络获取动作，在动作添加噪声然后将动作限制在有效的动作空间范围内
    # def get_actions(self, obs, add_noise):
    #     """Get actions for observations.
    #     Args:
    #         obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
    #         add_noise: (bool) whether to add noise
    #     Returns:
    #         actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
    #     """
    #     obs = check(obs).to(**self.tpdv)
    #     actions = self.actor(obs)
    #     if add_noise:
    #         actions += torch.randn_like(actions) * self.expl_noise * self.scale
    #         actions = torch.clamp(actions, self.low, self.high)
    #
    #     return actions
    ##离策略，继承自offpolicyBase类，HADDPG类实现深度确定性策略梯度算法异构版本
    ##类的构造函数，初始化类的实例，首先设置了一些基本参数
    def get_actions(self, state,add_noise):
        # 使用 Gumbel-Softmax 分布采样动作
        state = check(state).to(**self.tpdv)
        actions = self.actor(state)
        #device = logits.device

        #actions = gumbel_softmax_sample(logits, temperature=1.0, device=device)

        #actions = gumbel_softmax_sample(logits, temperature=1.0, device=torch.device("cuda"))
        #print(actions,"gum")
        #print(actions,logits)
        if np.random.random() < self.epsilon:
            action = torch.randint(low=0, high=25, size=(*state.shape[:-1], 1))
            #print(action,*state.shape[:-1],"test")
        else:
            #action_probs = torch.nn.functional.gumbel_softmax(actions, dim=-1, hard=True)
            action = actions.argmax(dim=-1, keepdim=True)
            #print(action,"tet")
            #print(action)
        #actions=action.tolist()
        #print(action, 'chazhao')

        #actions = torch.tensor(actions, dtype=torch.float32) * self.expl_noise
        action=action.tolist()
        action = torch.tensor(action, dtype=torch.float32) #* self.expl_noise

        action.requires_grad_()
        #print(action, 'chazhao')

        return action#action.item()

    def get_target_actions(self, obs):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        state = check(obs).to(**self.tpdv)
        actions = self.actor(state)
        # device=logits.device
        # actions = gumbel_softmax_sample(logits, temperature=1.0, device=device)
        # print(actions,"gum")
        if np.random.random() < self.epsilon:
            action = torch.randint(
                low=0, high=25, size=(*state.shape[:-1], 1)
            )
        else:
            action = actions.argmax(dim=-1, keepdim=True)
        action = action.tolist()
        action = torch.tensor(action, dtype=torch.float32) #* self.expl_noise
        action.requires_grad_()
        #action.requires_grad_()
        return action##action.item()
