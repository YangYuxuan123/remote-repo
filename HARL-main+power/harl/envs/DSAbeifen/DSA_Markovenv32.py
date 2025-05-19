import copy
import importlib
import logging
import random
import numpy as np
import supersuit as ss
from gym.spaces import Box
from gym.vector.utils import spaces
from matplotlib import pyplot as plt

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
from harl.envs.DSA.DSA_env5 import DSA_Markov

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
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 5
            self.args["max_cycles"] = 6
            # self.max_cycles = 25
            # self.args["max_cycles"] = 26

        self.cur_step = 0#设置当前步数初始值
        #self.module=importlib.import_module(''+self.scenario)
        #self.module = importlib.import_module("DSA." + self.scenario)#导入pettingzoo特定环境
        self.num_channels = 18+1#19#54#18##信道数#现在假设9channel，each agent观测3channel；但是又能观测整体channel
        self.num_channel = [19,19,19]#[8,8,8]#[18,18,18]#[6,6,6]
        self.num_agents = 3##num_agents  agent
        self.env = DSA_Markov(self.num_channel,self.num_channels, self.num_agents)
        self.env_copy = copy.deepcopy(self.env)

        self.sense_error_prob_max = 0.2
        self.sense_error_prob = np.random.uniform(0, self.sense_error_prob_max, size=(self.num_agents, self.num_channels))
        self.senselength = 4
        self.Dk = 2
        # Initialize the Markov channels
        self._build_Markov_channel()
        # Initialize the locations of SUs and PUs
        self._build_location()
        # Set the noise (mW)
        self.Noise = 1 * np.float_power(10, -8)
        # Set the carrier frequency (5 GHz)
        self.fc = 2.4
        # Set the K in channel gain
        self.K = 5
        # Set the power of PU and SU (mW)
        self.SU_power = 2#20
        self.PU_power = 20#40
        #self.env.reset()

        self.n_agents = self.num_agents
        #print("n_agents", self.n_agents)#3
        # self.agents = self.env.agents##当前活动智能体列表，每次环境实例中，只有部分智能体是活动的
        self.agents = ['agent_0', 'agent_1', 'agent_2']##当前活动智能体列表，每次环境实例中，只有部分智能体是活动的
        #print("agents", self.agents)##agents ['agent_0', 'agent_1', 'agent_2']
        # self.share_observation_space = self.repeat(self.env.state_space)

        # self.share_observation_space = [spaces.Box(self.low, self.high, (self.num_channel,), dtype=np.float32) for _ in range(self.num_agents)]
        #self.share_observation_space = [spaces.Discrete(2) for _ in range(self.num_agents)]
        # self.share_observation_space = [spaces.MultiBinary(self.num_channels) for _ in range(self.num_agents)]
        #self.share_observation_space =spaces.MultiBinary(self.num_channels)#np.random.choice(2, self.num_channels)
        # self.share_observation_space = self.repeat(spaces.MultiBinary(self.num_channels).sample())#self.repeat(self.env.state_space)
        #self.share_observation_space = self.repeat(np.random.choice(2, self.num_channels))#
        #self.share_observation_space=self.repeat(self.env.state_space)np.random.choice(2, self.n_channels)
        self.share_observation_space=self.repeat(self.env.get_state())
        # self.share_observation_space = self.repeat(self.env.get_state())#self.repeat(self.env.state_space)np.random.choice(2, self.n_channels)
        #print(self.share_observation_space,"23")
        #print(self.share_observation_space,"sha")

        # start_indices = np.cumsum([0] + self.num_channel[:-1])
        # end_indices = np.cumsum(self.num_channel)
        # # Get the observation for each agent
        # self.observation_space = [self.share_observation_space[start:end] for start, end in zip(start_indices, end_indices)]
        #[MultiBinary(9), MultiBinary(9), MultiBinary(9)]
        #print("share_observation_space", self.share_observation_space)##share_observation_space [Box(-inf, inf, (54,), float32), Box(-inf, inf, (54,), float32), Box(-inf, inf, (54,), float32)]
        # self.observation_space = [
        #     spaces.Box(low=0, high=1, shape=(6,), dtype=int) for _ in range(3)
        # ]

        # self.observation_spaces2= {'agent_0': spaces.MultiBinary(self.num_channel[0]).sample(),
        #                            'agent_1': spaces.MultiBinary(self.num_channel[1]).sample(),
        #                            'agent_2': spaces.MultiBinary(self.num_channel[2]).sample(),
        #                                              }
        #self.obs2 = [self.env.get_state()[0:8], self.env.get_state()[6:14], self.env.get_state()[10:18]]
        ##self.obs2 = self.env.get_obs()
        # self.obs2 = [self.env.get_state()[0:18], self.env.get_state()[0:18], self.env.get_state()[0:18]]
        #obs2 = [[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]

        self.observation_spaces2 = self.env.get_obs()
        #print(self.observation_spaces2,"2")
        # self.observation_spaces2= {'agent_0': self.share_observation_space[0:8],
        #                            'agent_1': self.share_observation_space[6:14],
        #                            'agent_2': self.share_observation_space[10:18],
        #                                              }
        #print(self.observation_spaces2,"2")
        self.observation_space = self.unwrap(self.observation_spaces2)
        #print(self.share_observation_space,"ssssss9")
        #print(self.observation_space,"obs")
        #self.observation_space = self.unwrap(self.env.observation_spaces)
        #print("observation_space", self.observation_space)##observation_space [Box(-inf, inf, (18,), float32), Box(-inf, inf, (18,), float32), Box(-inf, inf, (18,), float32)]#
        #0-6
        #self.action_spaces = {spaces.Discrete(self.num_channel + 1) for _ in range(self.num_agents)}
        #self.action_spaces = {f'agent_{i}': spaces.Discrete(self.num_channel + 1) for i in range(self.num_agents)}
        #self.action_spaces = {f'agent_{i}': spaces.Discrete(n + 1) for i, n in enumerate(self.num_channel)}
        # self.action_spaces = {f'agent_{i}': spaces.Discrete(n) for i, n in enumerate(self.num_channel)}
        #self.action_spaces = {f'agent_{i}': spaces.Discrete(n) for i, n in enumerate(self.num_channel)}
        #actiao=[]
        # action_ranges = [(0, 8), (6, 14), (10, 18)]  # List of action space ranges for each agent
        action_ranges = [(0, 18), (0, 18), (0, 18)]  # List of action space ranges for each agent
        self.action_spaces = {f'agent_{i}': np.append(np.arange(start, end), 18) for i, (start, end) in
                         enumerate(action_ranges)}
        self.all_actionspace=np.arange(self.num_channels)
        #self.action_spaces = {f'agent_{i}': np.arange(start, end) for i, (start, end) in enumerate(action_ranges)}

        #self.action_spaces = {f'agent_{i}':np.arange(n)for i, n in enumerate(self.num_channel)}

        #self.action_spaces = {}
        # for i in range(self.num_agents):
        #     start = random.randint(0, self.num_channels - self.num_channel[i])
        #     self.action_spaces[f'agent_{i}'] = list(range(start, start + self.num_channel[i]))
        #print(self.action_spaces,"ttttttttttttttt")
        #self.action_spaces={'agent_0': [12, 13, 14, 15, 16, 17], 'agent_1': [8, 9, 10, 11, 12, 13], 'agent_2': [0, 1, 2, 3, 4, 5]}
        self.action_space = self.unwrap(self.action_spaces)
        # for agent_id in range(self.num_agents):
        #     print(self.action_space[agent_id])
        #     print(type(self.action_space[agent_id]))
        #     if isinstance(self.action_space[agent_id], list):
        #         print("s")
        #print(self.action_spaces,"action_spaces")#{'agent_0': array([0, 1, 2, 3, 4, 5]), 'agent_1': array([ 6,  7,  8,  9, 10, 11]), 'agent_2': array([12, 13, 14, 15, 16, 17])} action_spaces
        # avail_actions = []
        # for agent_id in range(self.n_agents):
        #     avail_agent = self.get_avail_agent_actions(agent_id)
        #     avail_actions.append(avail_agent)
        # print(avail_actions,"avail_actions")

        #print(self.action_space,"act")

        # self.action_space = self.unwrap(self.env.action_spaces)
        #包含三个Box对象，每个环境中有三个智能体，每个智能体都有自己动作空间
        #print("action_space", self.action_space)##action_space [Box(0.0, 1.0, (5,), float32), Box(0.0, 1.0, (5,), float32), Box(0.0, 1.0, (5,), float32)]
        self._seed = 0
        #self.avail_actions = self.get_avail_actions()

    def _build_Markov_channel(self):
        # Initialize channel state (uniform distribution)
        # 1: Inactive state (primary user is not using)
        # 0: Active state (primary user is using)
        self.channel_state = np.random.choice(2, self.num_channels)
        # Initialize the transition probability of independent channels
        self.stayGood_prob = np.random.uniform(0.7, 1, self.num_channels)
        self.stayBad_prob = np.random.uniform(0, 0.3, self.num_channels)
        self.goodToBad_prob = 1 - self.stayGood_prob
        self.badToGood_prob = 1 - self.stayBad_prob

    def _build_location(self):
        # Initialize the location of PUs
        self.PU_TX_x = np.random.uniform(0, 150, self.num_channels)
        self.PU_TX_y = np.random.uniform(0, 150, self.num_channels)
        self.PU_RX_x = np.random.uniform(0, 150, self.num_channels)
        self.PU_RX_y = np.random.uniform(0, 150, self.num_channels)

        # Initialize the location of SUs transmitters
        self.SU_TX_x = np.random.uniform(0 + 40, 150 - 40, self.num_agents)
        self.SU_TX_y = np.random.uniform(0 + 40, 150 - 40, self.num_agents)

        # Initialize the distance between SUs' transmitter and receiver
        self.SU_d = np.random.uniform(20, 40, self.num_agents)

        # Initialize the location of SUs receivers
        SU_theda = 2 * np.pi * np.random.uniform(0, 1, self.num_agents)
        SU_dx = self.SU_d * np.cos(SU_theda)
        SU_dy = self.SU_d * np.sin(SU_theda)
        self.SU_RX_x = self.SU_TX_x + SU_dx
        self.SU_RX_y = self.SU_TX_y + SU_dy

        # Compute the distance between PU_TX and SU_RX
        self.SU_RX_PU_TX_d = np.zeros((self.num_agents, self.num_channels))
        for k in range(self.num_agents):
            for l in range(self.num_channels):
                self.SU_RX_PU_TX_d[k][l] = np.sqrt(
                    np.float_power(self.SU_RX_x[k] - self.PU_TX_x[l], 2) + np.float_power(
                        self.SU_RX_y[k] - self.PU_TX_y[l], 2))

        # Compute the distance between PU_TX and SU_RX
        self.SU_RX_SU_TX_d = np.zeros((self.num_agents, self.num_agents))
        for k1 in range(self.num_agents):
            for k2 in range(self.num_agents):
                self.SU_RX_SU_TX_d[k1][k2] = np.sqrt(
                    np.float_power(self.SU_RX_x[k1] - self.SU_TX_x[k2], 2) + np.float_power(
                        self.SU_RX_y[k1] - self.SU_TX_y[k2], 2))

        # Plot the locations
        # plt.plot(self.PU_TX_x, self.PU_TX_y, 'ro', label='PU_TX')
        # plt.plot(self.PU_RX_x, self.PU_RX_y, 'rx', label='PU_RX')
        # plt.plot(self.SU_TX_x, self.SU_TX_y, 'bs', label='SU_TX')
        # plt.plot(self.SU_RX_x, self.SU_RX_y, 'b^', label='SU_RX')
        # plt.legend(loc='lower right')
        # plt.ylabel('y')
        # plt.xlabel('x')
        # plt.show()

    # def sense(self):
    #     tmp_dice = np.random.uniform(0, 1, size=(self.num_agents, self.num_channels))  # roll the dice between 0 and 1
    #     error_index = tmp_dice < self.sense_error_prob # True: sensing error happens, False: sensing is correct
    #
    #     # Get the sensing result
    #     self.sensing_result = self.channel_state*(1-error_index) + (1-self.channel_state)*(error_index)
    #
    #     return self.sensing_result
    ##
    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """

        obs, rew, done, info = self.env.step(actions)
        # for action1 in actions:
        #     action1=[action[0] for action in action1]
        # obs, rew, done, info = self.env.step(actions)
        # print(obs,"w")#{'agent_0': array([0, 1, 0, 0, 0, 1]), 'agent_1': array([1, 1, 0, 0, 0, 0]), 'agent_2': array([0, 0, 1, 0, 0, 1])} w
        # #{'agent_0': array([[0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]]), 'agent_1': array([[0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]]), 'agent_2': array([[0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]])} w
        # print(rew,"s")#defaultdict(<class 'int'>, {'agent_0': array([-2.]), 'agent_1': array([-2.]), 'agent_2': array([-2.])}) s
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
        #print(obs,"obslast")
        # print(self.get_avail_actions(),"ava")#[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] ava
        # [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
        # ava
        #print(obs,"shaaa")
        #rewards = [[rew[0]]] * self.n_agents#
        # if self.img:
        #     obs = obs.transpose(0, 3, 1, 2)
        # return (
        #     self.split(obs),
        #     self.repeat(self.get_state()),
        #     rewards,
        #     self.repeat(done),
        #     self.repeat(info),
        #     self.avail_actions,
        # )
        return (
            self.unwrap(obs),
            s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )
    # def step(self, actions_batch):
    #     """
    #     return local_obs, global_state, rewards, dones, infos, available_actions
    #     """
    #     obs_batch=[]
    #     s_obs_batch=[]
    #     rew_batch=[]
    #     done_batch=[]
    #     info_batch=[]
    #     new_available_actionbatch=[]
    #     #print(actions,"actions")
    #     for actions in actions_batch:
    #         # Convert the list of actions to the correct format
    #         actions = [action[0] for action in actions]
    #         # Execute one step of the environment with the current actions
    #         obs, rew, done, info = self.env.step(actions)
    #
    #         self.cur_step += 1
    #         if self.cur_step == self.max_cycles:
    #             done = {agent: True for agent in self.agents}
    #             for agent in self.agents:
    #                 info[agent]['bad_transition'] = True
    #         dones = {agent: done[agent] for agent in self.agents}
    #         s_obs = self.repeat(self.env.get_state())
    #         # print(s_obs,"sobssss")
    #         total_reward = sum([rew[agent] for agent in self.agents])
    #         rewards = [[total_reward]] * self.n_agents
    #         obs_batch.append(self.unwrap(obs))
    #         rew_batch.append(rew)
    #         s_obs_batch.append(s_obs)
    #         done_batch.append(self.unwrap(dones))
    #         info_batch.append(self.unwrap(info))
    #         new_available_actionbatch.append(self.get_avail_actions())
    #
    #     # for action1 in actions:
    #     #     action1=[action[0] for action in action1]
    #     # obs, rew, done, info = self.env.step(actions)
    #     # print(obs,"w")#{'agent_0': array([0, 1, 0, 0, 0, 1]), 'agent_1': array([1, 1, 0, 0, 0, 0]), 'agent_2': array([0, 0, 1, 0, 0, 1])} w
    #     # #{'agent_0': array([[0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]]), 'agent_1': array([[0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]]), 'agent_2': array([[0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]])} w
    #     # print(rew,"s")#defaultdict(<class 'int'>, {'agent_0': array([-2.]), 'agent_1': array([-2.]), 'agent_2': array([-2.])}) s
    #     #print(obs,"obslast")
    #     # print(self.get_avail_actions(),"ava")#[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] ava
    #     # [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    #     # ava
    #     #print(obs,"shaaa")
    #     #rewards = [[rew[0]]] * self.n_agents#
    #     # if self.img:
    #     #     obs = obs.transpose(0, 3, 1, 2)
    #     # return (
    #     #     self.split(obs),
    #     #     self.repeat(self.get_state()),
    #     #     rewards,
    #     #     self.repeat(done),
    #     #     self.repeat(info),
    #     #     self.avail_actions,
    #     # )
    #     return (
    #         obs_batch,
    #         s_obs_batch,
    #         rew_batch,
    #         done_batch,
    #         info_batch,
    #         new_available_actionbatch,
    #     )

    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1  ##增加随机种子的值，随机种子用于控制环境的随机性，以便在需要的时候重现相同的环境状态
        self.cur_step = 0  ##这行代码将当前步数重置为0。这是因为每次开始新的回合时，都需要从第0步开始。
        obs = self.unwrap(self.observation_spaces2)  #
        #self.repeat(spaces.MultiBinary(self.num_channels).sample())
        # s_obs = self.repeat(self.env.get_state())
        s_obs = self.repeat(self.env.get_state())#(self.repeat(spaces.MultiBinary(self.num_channels).sample()))
        #print(s_obs,"+++++++++++++")#出现各种情况？

        return obs, s_obs, self.get_avail_actions()  #

    # def get_avail_actions(self):
    #     if self.discrete:
    #         avail_actions = []
    #         for agent_id in range(self.n_agents):
    #             avail_agent = self.get_avail_agent_actions(agent_id)
    #             avail_actions.append(avail_agent)
    #         return avail_actions
    #     else:
    #         return None
    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        #print(avail_actions,"avvv")
        return avail_actions
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        #return [1] * self.action_space[agent_id].n
        return self.action_space[agent_id]

    def render(self):
       self.env.render()

    def renders(self):
        # The probability of staying in current state in next time slot
        stay_prob = self.channel_state*self.stayGood_prob + (1-self.channel_state)*self.stayBad_prob
        tmp_dice = np.random.uniform(0, 1, self.n_channel) # roll the dice between 0 and 1
        stay_index = tmp_dice < stay_prob # 1: stay in current state, 0: change state

        # Update the channel state
        self.channel_state = self.channel_state*stay_index + (1-self.channel_state)*(1-stay_index)

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