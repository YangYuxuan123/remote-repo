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
        ##设置环境的最大周期数max_cycles，这个数字是在环境中随机生成的，但是我这边设置为25，也就是说环境中的周期数是25
        #当达到最大周期数时环境会自动重置
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 5
            self.args["max_cycles"] = 6
        self.senselength = 8  # 感知长度
        self.Dk = 3
        self.cur_step = 0#设置当前步数初始值
        self.num_channels =25#25 #25#18#25#18+self.senselength-1#19#54#18##信道数#现在假设9channel，each agent观测3channel；但是又能观测整体channel
        self.num_channel = [25,25,25]#[25,25,25]#[18,18,18]#[8,8,8]#[18,18,18]#[6,6,6]
        self.num_agents = 3##num_agents  agent
        self.env = DSA_Markov(self.num_channel,self.num_channels, self.num_agents)
        self.env_copy = copy.deepcopy(self.env)

        self.sense_error_prob_max = 0.2
        self.sense_error_prob = np.random.uniform(0, self.sense_error_prob_max, size=(self.num_agents, self.num_channels))
        # Initialize the Markov channels
        # self._build_Markov_channel()
        # # Initialize the locations of SUs and PUs
        # self._build_location()
        # Set the noise (mW)
        self.Noise = 1 * np.float_power(10, -8)
        # Set the carrier frequency (5 GHz)
        self.fc = 2.4
        # Set the K in channel gain
        self.K = 5
        # Set the power of PU and SU (mW)
        self.SU_power = 20#20
        self.PU_power = 40#40
        #self.env.reset()

        self.n_agents = self.num_agents
        #print("n_agents", self.n_agents)#3
        # self.agents = self.env.agents##当前活动智能体列表，每次环境实例中，只有部分智能体是活动的
        self.agents = ['agent_0', 'agent_1', 'agent_2']##当前活动智能体列表，每次环境实例中，只有部分智能体是活动的
        self.share_observation_spaces = {
            'agent_0': MultiBinary(32),
            'agent_1': MultiBinary(32),
            'agent_2': MultiBinary(32),
        }
        self.share_observation_space=self.unwrap(self.share_observation_spaces)
        #self.observation_spaces2 = self.env.get_obs()
        self.observation_spaces = {
            'agent_0': MultiBinary(32),
            'agent_1': MultiBinary(32),
            'agent_2': MultiBinary(32),
        }
        self.observation_space = self.unwrap(self.observation_spaces)
        self.action_spaces = {
            'agent_0': Discrete(25),
            'agent_1': Discrete(25),
            'agent_2': Discrete(25),
        }
        ##self.all_actionspace=np.arange(self.num_channels+1)
        #print(self.action_spaces,"2")
        self.action_space = self.unwrap(self.action_spaces)
        self._seed = 0
        #self.avail_actions = self.get_avail_actions()


    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        self.success_hist_1 = []
        self.fail_collision_hist_1 = []
        self.fail_PU_hist_1 = []


        # print(actions,'test')
        self.success = 0
        self.fail_PU = 0
        self.fail_collision = 0
        actions= actions.astype(int)
        actions = [item for sublist in actions for item in sublist]
        self.env.render()
        self.env.render_SINR()
        obs, rew, done, info = self.env.step(actions)
        self.success_hist_1 = self.env.success
        self.fail_collision_hist_1 = self.env.fail_collision
        self.fail_PU_hist_1=self.env.fail_PU
        #print(obs,"tets")
        self.cur_step+=1
        if self.cur_step==self.max_cycles:
            done = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]['bad_transition']=True
        dones={agent:done[agent] for agent in self.agents}
        s_obs=self.repeat(self.env.get_state())
        #print(s_obs,"sobssss")
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
        # print(rewards)#[[-5.070866175941307], [-5.070866175941307], [-5.070866175941307]]
        # print(self.fail_collision_hist_1)2
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
        self._seed += 1  ##增加随机种子的值，随机种子用于控制环境的随机性，以便在需要的时候重现相同的环境状态
        self.cur_step = 0  ##这行代码将当前步数重置为0。这是因为每次开始新的回合时，都需要从第0步开始。
        #obs = self.unwrap(self.observation_spaces)  #
        obs = self.unwrap(self.env.get_obs())
        ##obs=self.repeat(obs)
        ##print(obs,"obs")
        ##print(obs,"kjj")
        s_obs = self.repeat(self.env.get_state())#(self.repeat(spaces.MultiBinary(self.num_channels).sample()))
        #print(s_obs,"+++++++++++++")#出现各种情况？
        return obs, s_obs, self.get_avail_actions()  #

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        #print(avail_actions,"avvv")
        return avail_actions
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n
        #return self.action_space[agent_id]

    def render(self):
       self.env.render()

    # def render_SINR(self):
    #     # Update the SINR
    #
    #     # Calculate the channel gain
    #     SU_d = copy.deepcopy(np.reshape(self.SU_d, (-1, 1)))
    #     for n in range(self.n_channel-1):
    #         SU_d = np.hstack( (SU_d, np.reshape(self.SU_d, (-1, 1))) )
    #
    #     SU_sigma2 = np.float_power(10, -((41+22.7*np.log10(SU_d)+20*np.log10(self.fc/5))/10))
    #     CN_real = np.random.normal(0, 1, size=(self.num_agents, self.n_channel))
    #     CN_imag = np.random.normal(0, 1, size=(self.num_agents, self.n_channel))
    #     theda = np.random.uniform(0, 1, size=(self.num_agents, self.n_channel))
    #     H = np.sqrt(self.K/(self.K+1)*SU_sigma2)*np.exp(1j*2*np.pi*theda) + np.sqrt(1/(self.K+1)*SU_sigma2/2)*(CN_real + 1j*CN_imag)
    #     self.H2 = np.float_power(np.absolute(H), 2)
    #
    #     # Calculate the interference of PUs
    #     PU_sigma2 = np.float_power(10, -((41 + 22.7 * np.log10(self.SU_RX_PU_TX_d) + 20 * np.log10(self.fc / 5)) / 10))
    #     channel_state = np.array([self.channel_state for k in range(self.num_agents)])
    #     self.Interferecne_PU = self.PU_power * PU_sigma2 * (1 - channel_state)
    #
    #     self.SINR = self.H2*self.SU_power/(self.Interferecne_PU + self.Noise)
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
    # def get_state(self):
    #     # adapted from imple115StateWrapper.convert_observation
    #     raw_state = self.env.unwrapped.observation()
    #     def do_flatten(obj):
    #         """Run flatten on either python list or numpy array."""
    #         if type(obj) == list:
    #             return np.array(obj).flatten()
    #         return obj.flatten()
    #
    #     s = []
    #     for i, name in enumerate(
    #         ["left_team", "left_team_direction", "right_team", "right_team_direction"]
    #     ):
    #         s.extend(do_flatten(raw_state[0][name]))
    #         # If there were less than 11vs11 players we backfill missing values
    #         # with -1.
    #         if len(s) < (i + 1) * 22:
    #             s.extend([-1] * ((i + 1) * 22 - len(s)))
    #     # ball position
    #     s.extend(raw_state[0]["ball"])
    #     # ball direction
    #     s.extend(raw_state[0]["ball_direction"])
    #     # one hot encoding of which team owns the ball
    #     if raw_state[0]["ball_owned_team"] == -1:
    #         s.extend([1, 0, 0])
    #     if raw_state[0]["ball_owned_team"] == 0:
    #         s.extend([0, 1, 0])
    #     if raw_state[0]["ball_owned_team"] == 1:
    #         s.extend([0, 0, 1])
    #     game_mode = [0] * 7
    #     game_mode[raw_state[0]["game_mode"]] = 1
    #     s.extend(game_mode)
    #     for obs in raw_state:
    #         active = [0] * 11
    #         if obs["active"] != -1:
    #             active[obs["active"]] = 1
    #         s.extend(active)
    #     return np.array(s, dtype=np.float32)

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
                    low=obs_sp.low[idx],
                    high=obs_sp.high[idx],
                    shape=obs_sp.shape[1:],
                    dtype=obs_sp.dtype,
                )
                for idx in range(self.n_agents)
            ]

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

    def split(self, a):
        return [a[i] for i in range(self.n_agents)]
    ##是一个方法，返回一个列表，其中包含每个智能体的可用动作；
    #如果动作空间是离散的（self.discrete 为 True），那么对于每个智能体，
    # 它都会调用 get_avail_agent_actions 方法来获取该智能体的可用动作，
    # 然后将这些动作添加到 avail_actions 列表中。
    ##如果动作空间不是离散的（self.discrete 为 False），那么它将返回 None。
    ##假设存在的方法，接收一个智能体的ID，然后返回这个智能体的可用动作，需要根据环境和智能体的实现定义方法
    #get_avail_actions 方法返回 avail_actions，这是一个列表，其中每个元素都是一个智能体的可用动作。
    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions