import copy
import importlib
import logging
import random
import numpy as np
import pandas as pd
import supersuit as ss
from gym.spaces import Box, MultiBinary
from gym.vector.utils import spaces
from gym.spaces import Discrete
from matplotlib import pyplot as plt

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

from harl.envs.DSA.DSA_env import DSA_Markov

data_in = pd.read_csv("./real_data_trace.csv")
data_in = data_in.drop("index",axis=1)

class DSA_MarkovEnv:
    def __init__(self, args):
        random_seed=1
        np.random.seed(random_seed)
        self.args = copy.deepcopy(args)
        self.scenario = args["scenario"]
        del self.args["scenario"]
        self.discrete = True##动作离散
        if (
            "continuous_actions" in self.args
            and self.args["continuous_actions"] == True
        ):
            self.discrete = False
        ##环境的最大周期数 max_cycles
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 5
            self.args["max_cycles"] = 6
        self.senselength = 8
        self.Dk = 4
        self.cur_step = 0
        self.num_channels = 25 #总信道有32个(从0开始算)，聚合信道有25个(从0开始算)
        self.num_channel = [25,25,25]
        self.num_agents = 3
        self.env = DSA_Markov(self.num_channel,self.num_channels, self.num_agents)
        self.env_copy = copy.deepcopy(self.env)

        self.sense_error_prob_max = 0.2
        self.sense_error_prob = np.random.uniform(0, self.sense_error_prob_max, size=(self.num_agents, self.num_channels))

        self.SU_power = 20
        self.PU_power = 40
        self.n_agents = self.num_agents

        self.agents = ['agent_0', 'agent_1', 'agent_2']##当前活动智能体列表
        self.share_observation_spaces = {
            'agent_0': MultiBinary(self.num_channels+self.senselength-1),
            'agent_1': MultiBinary(self.num_channels+self.senselength-1),
            'agent_2': MultiBinary(self.num_channels+self.senselength-1),
        }
        self.share_observation_space=self.unwrap(self.share_observation_spaces)
        self.observation_spaces = {
            'agent_0': MultiBinary(self.senselength),
            'agent_1': MultiBinary(self.senselength),
            'agent_2': MultiBinary(self.senselength),
        }
        self.observation_space = self.unwrap(self.observation_spaces)
        self.action_spaces = {
            'agent_0': Discrete(self.num_channels+1),
            'agent_1': Discrete(self.num_channels+1),
            'agent_2': Discrete(self.num_channels+1),
        }

        self.action_space = self.unwrap(self.action_spaces)
        self._seed = 0

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        self.success_hist_1 = [] #存储每个智能体的成功状态序列
        self.fail_collision_hist_1 = [] #存储碰撞失败序列
        self.fail_PU_hist_1 = [] #存储主用户冲突失败序列

        self.success = 0 #当前步成功次数
        self.fail_PU = 0 #当前步与主用户冲突失败次数
        self.fail_collision = 0 #当前步智能体间碰撞失败次数
        actions= actions.astype(int)
        actions = [item for sublist in actions for item in sublist]
        self.env.render()
        self.env.render_SINR()
        obs, rew, done, info = self.env.step(actions)
        self.success_hist_1 = self.env.success
        self.fail_collision_hist_1 = self.env.fail_collision
        self.fail_PU_hist_1=self.env.fail_PU

        self.cur_step+=1
        if self.cur_step==self.max_cycles:
            done = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]['bad_transition'] = True
        dones = {agent:done[agent] for agent in self.agents}
        s_obs = self.repeat(self.env.get_state())

        total_reward = sum([rew[agent] for agent in self.agents])
        rewards=[[total_reward]]*self.n_agents
        self.success = self.env.success
        self.fail_PU = self.env.fail_PU
        self.fail_collision1 = np.array(self.env.fail_collision)
        self.fail_collision = self.fail_collision1.reshape(-1, 1)
        self.success_hist1 = np.array(self.env.success)
        self.success_hist = self.success_hist1.reshape(-1, 1)
        self.fail_PU_hist1=np.array(self.env.fail_PU)
        self.fail_PU_hist=self.fail_PU_hist1.reshape(-1,1)
        #假设有3个智能体，在某个步骤中，两个成功，一个失败，那么success_hist_1可能记录为[True, True, False]，而success计数器为2
        #将任意形状的数值/数组转换为二维列向量

        return (
            self.unwrap(obs),
            s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
            self.success_hist,
            self.fail_collision,
            self.fail_PU_hist
        )

    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1
        self.cur_step = 0
        obs = self.unwrap(self.env.get_obs([0,0,0]))
        s_obs = self.repeat(self.env.get_state())
        return obs, s_obs, self.get_avail_actions()  #

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)

        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed

    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

    def get_obs_shape(self):
        obs_sp = self.env.observation_space
        if self.img:
            w, h, c = self.env.observation_space.shape[1:]
            return [
                Box(
                    low=obs_sp.low[idx].transpose(2, 0, 1),
                    high=obs_sp.high[idx].transpose(2, 0, 1),
                    shape=(c, w, h),
                    dtype=obs_sp.dtype,
                )
                for idx in range(self.n_agents)
            ]
        else:
            return [
                Box(
                    low = obs_sp.low[idx],
                    high = obs_sp.high[idx],
                    shape = obs_sp.shape[1:],
                    dtype = obs_sp.dtype,
                )
                for idx in range(self.n_agents)
            ]

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

    def split(self, a):
        return [a[i] for i in range(self.n_agents)]

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions