from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import copy

from gym import spaces
random_seed=1
np.random.seed(random_seed)

class DSA_Markov():
    def __init__(
            self,
            nc_all,
            n_channels,
            num_agents,
            sense_error_prob_max = 0.1,
            punish_interfer_PU = -5
    ):
        self.cur_step = 0#设置当前步数初始值
        self.nc_all=nc_all
        self.n_channels = n_channels # The number of the channels
        self.senselength = 10  # 感知长度
        self.Dk = 5
        self.num_agents = num_agents # The number of the SUs
        ####依据情况更改
        self._has_reset=False
        self._has_rendered = False
        self._has_updated = False

        # Initialize the Markov channels
        self._build_Markov_channel()

        # Initialize the locations of SUs and PUs
        self._build_location()

        # Set the noise (mW)
        self.Noise = 1 * np.float_power(10, -8)
        # Set the carrier frequency (5 GHz)
        self.fc = 5
        # Set the K in channel gain
        self.K = 5
        # Set the power of PU and SU (mW)
        self.SU_power = 20
        self.PU_power = 40

        # Initialize SINR (no consider interference of SUs)
        self.render_SINR()

        self.n_actions = n_channels + 1 # The action space size
        self.n_features = n_channels # The sensing result space


        self.sense_error_prob_max = sense_error_prob_max
        self.sense_error_prob = np.random.uniform(0, self.sense_error_prob_max, size=(self.num_agents, self.n_channels+self.senselength-1))
        self.sense_error_probagent1 = np.random.uniform(0, self.sense_error_prob_max, size=(self.num_agents, self.nc_all[0]))
        self.sense_error_prob_agg = np.random.uniform(0, self.sense_error_prob_max, size=(self.num_agents, self.n_channels-self.senselength+1))

        # The punishment for interfering PUs
        self.punish_interfer_PU = punish_interfer_PU
        #self.state_space=np.random.choice(2, self.n_channel)
        self.F = list(range(self.n_channels+1))  # 其中包含了频道的索引。这些频道索引表示了可用频道的标识。在代码中，self.F 的初始化使用了 range(num_channels)，它将创
        self.P = [0.1, 0.5, 1.0]
        self._seed = 0
    def reset(self, seed=None, return_info=False, options=None):
        self._has_reset = True
        self._has_updated = True
        self.seed=None
        self.options=None
        #super().reset(seed=seed, options=options)

    def _build_Markov_channel(self):

        self.channel_state = np.random.choice(2, self.n_channels+self.senselength-1)

        self.obs_block = [self.channel_state[i:i + self.senselength] for i in range(self.n_channels)]
        # 判断带宽要求:满足可传输要求
        # print(obs)
        # print("++++++++++++++++++++")
        self.channel_position = [list(range(i, i + self.senselength)) for i in range(self.n_channels)]

        self.channelagg_state = [1 if sum(1 for bit in observation if bit == 1) >= self.Dk else 0 for observation in
                                 self.obs_block]

        self.stayGood_prob = np.random.uniform(0.7, 1, self.n_channels + self.senselength - 1)
        self.stayBad_prob = np.random.uniform(0, 0.3, self.n_channels + self.senselength - 1)
        self.goodToBad_prob = 1 - self.stayGood_prob
        self.badToGood_prob = 1 - self.stayBad_prob
        ##agg
        self.stayGood_prob_agg = np.random.uniform(0.7, 1, self.n_channels)
        self.stayBad_prob_agg = np.random.uniform(0, 0.3, self.n_channels)
        self.goodToBad_prob_agg = 1 - self.stayGood_prob_agg
        self.badToGood_prob_agg = 1 - self.stayBad_prob_agg

    # def _build_location(self):
    #
    #     # Initialize the location of PUs
    #     self.PU_TX_x = np.random.uniform(0, 150, self.n_channels)
    #     self.PU_TX_y = np.random.uniform(0, 150, self.n_channels)
    #     self.PU_RX_x = np.random.uniform(0, 150, self.n_channels)
    #     self.PU_RX_y = np.random.uniform(0, 150, self.n_channels)
    #
    #     # Initialize the location of SUs transmitters
    #     self.SU_TX_x = np.random.uniform(0+40, 150-40, self.num_agents)
    #     self.SU_TX_y = np.random.uniform(0+40, 150-40, self.num_agents)
    #
    #     # Initialize the distance between SUs' transmitter and receiver
    #     self.SU_d = np.random.uniform(20, 40, self.num_agents)
    #
    #     # Initialize the location of SUs receivers
    #     SU_theda = 2 * np.pi * np.random.uniform(0, 1, self.num_agents)
    #     SU_dx = self.SU_d * np.cos(SU_theda)
    #     SU_dy = self.SU_d * np.sin(SU_theda)
    #     self.SU_RX_x = self.SU_TX_x + SU_dx
    #     self.SU_RX_y = self.SU_TX_y + SU_dy
    #
    #     # Compute the distance between PU_TX and SU_RX
    #     self.SU_RX_PU_TX_d = np.zeros((self.num_agents, self.n_channels))
    #     for k in range(self.num_agents):
    #         for l in range(self.n_channels):
    #             self.SU_RX_PU_TX_d[k][l] = np.sqrt(
    #                 np.float_power(self.SU_RX_x[k] - self.PU_TX_x[l], 2) + np.float_power(
    #                     self.SU_RX_y[k] - self.PU_TX_y[l], 2))
    #
    #     # Compute the distance between PU_TX and SU_RX
    #     self.SU_RX_SU_TX_d = np.zeros((self.num_agents, self.num_agents))
    #     for k1 in range(self.num_agents):
    #         for k2 in range(self.num_agents):
    #             self.SU_RX_SU_TX_d[k1][k2] = np.sqrt(
    #                 np.float_power(self.SU_RX_x[k1] - self.SU_TX_x[k2], 2) + np.float_power(
    #                     self.SU_RX_y[k1] - self.SU_TX_y[k2], 2))

    # def _build_locationagent1(self):
    #
    #     # Initialize the location of PUs
    #     self.PU_TX_x = np.random.uniform(0, 150, self.nc_all[0])
    #     self.PU_TX_y = np.random.uniform(0, 150, self.nc_all[0])
    #     self.PU_RX_x = np.random.uniform(0, 150, self.nc_all[0])
    #     self.PU_RX_y = np.random.uniform(0, 150, self.nc_all[0])
    #
    #     # Initialize the location of SUs transmitters
    #     self.SU_TX_x = np.random.uniform(0+40, 150-40, self.num_agents)
    #     self.SU_TX_y = np.random.uniform(0+40, 150-40, self.num_agents)
    #
    #     # Initialize the distance between SUs' transmitter and receiver
    #     self.SU_d = np.random.uniform(20, 40, self.num_agents)
    #
    #     # Initialize the location of SUs receivers
    #     SU_theda = 2 * np.pi * np.random.uniform(0, 1, self.num_agents)
    #     SU_dx = self.SU_d * np.cos(SU_theda)
    #     SU_dy = self.SU_d * np.sin(SU_theda)
    #     self.SU_RX_x = self.SU_TX_x + SU_dx
    #     self.SU_RX_y = self.SU_TX_y + SU_dy
    #
    #     # Compute the distance between PU_TX and SU_RX
    #     self.SU_RX_PU_TX_d = np.zeros((self.num_agents, self.nc_all[0]))
    #     for k in range(self.num_agents):
    #         for l in range(self.nc_all[0]):
    #             self.SU_RX_PU_TX_d[k][l] = np.sqrt(
    #                 np.float_power(self.SU_RX_x[k] - self.PU_TX_x[l], 2) + np.float_power(
    #                     self.SU_RX_y[k] - self.PU_TX_y[l], 2))
    #
    #     # Compute the distance between PU_TX and SU_RX
    #     self.SU_RX_SU_TX_d = np.zeros((self.num_agents, self.num_agents))
    #     for k1 in range(self.num_agents):
    #         for k2 in range(self.num_agents):
    #             self.SU_RX_SU_TX_d[k1][k2] = np.sqrt(
    #                 np.float_power(self.SU_RX_x[k1] - self.SU_TX_x[k2], 2) + np.float_power(
    #                     self.SU_RX_y[k1] - self.SU_TX_y[k2], 2))
    #
    #     # Plot the locations
    #     # plt.plot(self.PU_TX_x, self.PU_TX_y, 'ro', label='PU_TX')
    #     # plt.plot(self.PU_RX_x, self.PU_RX_y, 'rx', label='PU_RX')
    #     # plt.plot(self.SU_TX_x, self.SU_TX_y, 'bs', label='SU_TX')
    #     # plt.plot(self.SU_RX_x, self.SU_RX_y, 'b^', label='SU_RX')
    #     # plt.legend(loc='lower right')
    #     # plt.ylabel('y')
    #     # plt.xlabel('x')
    #     # plt.show()
    def _build_location(self):

        # Initialize the location of PUs
        self.PU_TX_x = np.random.uniform(0, 200, self.n_channels + self.senselength - 1)
        self.PU_TX_y = np.random.uniform(0, 200, self.n_channels + self.senselength - 1)
        self.PU_RX_x = np.random.uniform(0, 200, self.n_channels + self.senselength - 1)
        self.PU_RX_y = np.random.uniform(0, 200, self.n_channels + self.senselength - 1)
        # self.PU_TX_x = np.random.uniform(0, 150, self.n_channel)
        # self.PU_TX_y = np.random.uniform(0, 150, self.n_channel)
        # self.PU_RX_x = np.random.uniform(0, 150, self.n_channel)
        # self.PU_RX_y = np.random.uniform(0, 150, self.n_channel)

        # Initialize the location of SUs transmitters
        self.SU_TX_x = np.random.uniform(0+40, 200-40, self.num_agents)
        self.SU_TX_y = np.random.uniform(0+40, 200-40, self.num_agents)
        # self.SU_TX_x = np.random.uniform(0+40, 20-40, self.n_su)
        # self.SU_TX_y = np.random.uniform(0+40, 20-40, self.n_su)

        # Initialize the distance between SUs' transmitter and receiver
        self.SU_d = np.random.uniform(20, 100, self.num_agents)

        # Initialize the location of SUs receivers
        SU_theda = 2 * np.pi * np.random.uniform(0, 1, self.num_agents)
        SU_dx = self.SU_d * np.cos(SU_theda)
        SU_dy = self.SU_d * np.sin(SU_theda)
        self.SU_RX_x = self.SU_TX_x + SU_dx


        self.SU_RX_y = self.SU_TX_y + SU_dy

        # Compute the distance between PU_TX and SU_RX
        self.SU_RX_PU_TX_d = np.zeros((self.num_agents, self.n_channels + self.senselength - 1))
        for k in range(self.num_agents):
            for l in range(self.n_channels + self.senselength - 1):
                self.SU_RX_PU_TX_d[k][l] = np.sqrt(
                    np.float_power(self.SU_RX_x[k] - self.PU_TX_x[l], 2) + np.float_power(
                        self.SU_RX_y[k] - self.PU_TX_y[l], 2))

        # Compute the distance between SU_TX and SU_RX
        self.SU_RX_SU_TX_d = np.zeros((self.num_agents, self.num_agents))
        for k1 in range(self.num_agents):
            for k2 in range(self.num_agents):
                self.SU_RX_SU_TX_d[k1][k2] = np.sqrt(
                    np.float_power(self.SU_RX_x[k1] - self.SU_TX_x[k2], 2) + np.float_power(
                        self.SU_RX_y[k1] - self.SU_TX_y[k2], 2))

        # #Plot the locations

        # plt.plot(self.PU_TX_x, self.PU_TX_y, 'ro', label='PU_TX')
        # plt.plot(self.PU_RX_x, self.PU_RX_y, 'rx', label='PU_RX')
        # plt.plot(self.SU_TX_x, self.SU_TX_y, 'bs', label='SU_TX')
        # plt.plot(self.SU_RX_x, self.SU_RX_y, 'b^', label='SU_RX')
        # plt.legend(loc='lower right')
        # plt.ylabel('y')
        # plt.xlabel('x')
        # plt.show()

    def close(self):
        """Closes the rendering window."""
        pass
    def store_action(self, action):
        self.action = action

    def get_state(self):

        self.getstate=self.channel_state
        self.obs_block = [self.channel_state[i:i + self.senselength] for i in range(self.n_channels)]
        # 判断带宽要求:满足可传输要求
        # print(obs)
        # print("++++++++++++++++++++")
        self.channel_position = [list(range(i, i + self.senselength)) for i in range(self.n_channels)]

        self.channelagg_state = [1 if sum(1 for bit in observation if bit == 1) >= self.Dk else 0 for observation in
                                 self.obs_block]
        #getstate=self.channel_state
        return self.getstate
    def sense(self):
        tmp_dice = np.random.uniform(0, 1, size=(self.num_agents, self.n_channels+self.senselength-1))  # roll the dice between 0 and 1
        error_index = tmp_dice < self.sense_error_prob # True: sensing error happens, False: sensing is correct

        # Get the sensing result
        self.sensing_result = self.get_state()*(1-error_index) + (1-self.get_state())*(error_index)

        return self.sensing_result
    def get_obs(self):
        obs2=self.sense()
        agents = ['agent_0', 'agent_1', 'agent_2']
        obs = {agent: observation for agent, observation in zip(agents, obs2)}

        return obs

    def step(self, action):
        self.success = 0
        self.fail_PU = 0
        self.fail_collision = 0
        #print(action,'h')
        #action=action.astype(int)
        #print(action,'chang')
        #rewards = np.zeros(self.num_agents)
        obs = self.get_obs()#[array([0*, 0, 0, 1, 1, 1]), array([0, 1, 0, 0, 0, 0]), array([1, 0, 0, 1, 1, 1])]

        self.render()
        self.render_SINR()
        rewards = np.zeros((self.num_agents))
        success=np.zeros((self.num_agents))
        #mrewards= np.zeros((self.num_agents))
        #self.reward = np.zeros(self.n_su)

        # Calculate the interference of SUs
        Interferecne_SU = 0
        less_than_8 = self.SU_RX_SU_TX_d < 8
        SU_sigma2 = np.where(less_than_8, np.float_power(10, -((32.45 + 20 * np.log10(self.fc * self.SU_RX_SU_TX_d)) / 10)),
                             np.float_power(10, -((58.3 + 33 * np.log10(self.SU_RX_SU_TX_d / 8)) / 10)))
        #SU_sigma2 = np.float_power(10, -((41 + 22.7 * np.log10(self.SU_RX_SU_TX_d) + 20 * np.log10(self.fc / 5)) / 10))
        # SU_sigma2=np.float_power(10, -((32.45+20*np.log10(self.fc*self.SU_RX_SU_TX_d))/10))
        #                                  # + 20 * np.log10(self.fc / 5)) / 10))

        for k in range(self.num_agents):
            SU_sigma2[k][k] = 0
        #print(action,"ac")
        for k in range(self.num_agents):#0,1,2
            if (action[k] == self.n_channels) :#or self.channelagg_state[k]==0: # action is not choosing any channel
                rewards[k] = -2
                success[k]=0
            else: # action is choosing one of channel blocks
                for q in range(self.num_agents):
                    # if (action[q] == action[k]):
                    def check_overlap_and_ones(pos1, pos2, channel_state):
                        # 转换为集合并找出重叠部分
                        overlap = set(pos1).intersection(set(pos2))
                        # 找出不重叠部分的位置
                        non_overlap_pos1 = [pos for pos in pos1 if pos not in overlap]
                        non_overlap_pos2 = [pos for pos in pos2 if pos not in overlap]
                        # 找出不重叠部分的数据
                        non_overlap_data1 = [channel_state[pos] for pos in non_overlap_pos1]
                        non_overlap_data2 = [channel_state[pos] for pos in non_overlap_pos2]
                        # 如果不重叠部分数据为1的数量大于等于3
                        if sum(non_overlap_data1) >= self.Dk or sum(non_overlap_data2) >= self.Dk or not overlap:
                            return False
                        else:
                            return True
                    if action[q]!=self.n_channels:
                        if check_overlap_and_ones(self.channel_position[action[k]],self.channel_position[action[q]],self.channel_state):
                            Interferecne_SU = Interferecne_SU + SU_sigma2[k][q] * self.SU_power
                    # if action[q]!=self.n_channels-self.senselength+1:
                    #     #print(action)
                    #     # print(self.channel_position[action[k]])
                    #     # print(self.channel_state)
                    #     if check_overlap_and_ones(self.channel_position[action[k]],self.channel_position[action[q]],self.channel_state):
                    #         Interferecne_SU = Interferecne_SU + SU_sigma2[k][q] * self.SU_power
                        # success[k]=success[k]
                index_channelblock = action[k]
                cb=self.channel_position[index_channelblock]
                x_f=0
                y_f=0
                for x in range(len(cb)):
                    x_f = x_f + self.H2[k, cb[x]] * self.SU_power
                    y_f = y_f + self.Interferecne_PU[k, cb[x]]
                    #SINR=SINR+self.channel_block[k][x]*self.H2[k, cb[x]] * self.SU_power/ (Interferecne_SU + self.Interferecne_PU[k, cb[x]] + self.Noise)
                SINR=x_f/(Interferecne_SU+y_f+self.Noise*self.Dk)
                # SINR = self.H2[k, action[k]] * self.SU_power / (
                #                 Interferecne_SU + self.Interferecne_PU[k, action[k]] + self.Noise)
                    # self.reward[k] = np.log2(1 + SINR)
                #SINR = self.H2[k, action[k]] * self.SU_power / (Interferecne_SU + self.Interferecne_PU[k, action[k]] + self.Noise)
                #SINR = self.H2[k, action[k]] * self.SU_power / (Interferecne_SU + self.Interferecne_PU[k, action[k]] + self.Noise)
                rewards[k] = np.log2(1 + SINR)
                success[k]=success[k]+1

                def check_overlap_and_ones2(pos2, channel_state, channel_blockposition, action):
                    for i in range(len(action)):
                        if (action[i] != self.n_channels):
                            pos1 = channel_blockposition[action[i]]
                            if pos1 != pos2:
                                overlap = set(pos1).intersection(set(pos2))
                                non_overlap_pos1 = [pos for pos in pos1 if pos not in overlap]
                                non_overlap_pos2 = [pos for pos in pos2 if pos not in overlap]
                                non_overlap_data1 = [channel_state[pos] for pos in non_overlap_pos1]
                                non_overlap_data2 = [channel_state[pos] for pos in non_overlap_pos2]
                                if sum(non_overlap_data2) < self.Dk:
                                    return False,sum(non_overlap_data2)
                    return True,sum(pos1)
                if (self.channelagg_state[action[k]] == 1):
                    # if (len(np.where(action == action[k])[0]) == 1 ):
                    panduan,bili=check_overlap_and_ones2(self.channel_position[action[k]], self.channel_state, self.channel_position, action)
                    if (panduan):
                        # successful tr/ansmission##成功传输
                        self.success = self.success + 1
                        #success[k] = success[k]+self.senselength
                        rewards[k]=rewards[k]
                    else:
                        #print("no")
                        self.fail_collision = self.fail_collision+ 1
                        success[k] = 1
                        success[k] = success[k]-self.Dk
                        rewards[k]=rewards[k]*((((self.Dk-bili)/self.num_agents)+(self.Dk-(self.Dk-bili)))/(self.Dk))
                        #rewards[k]=rewards[k]+self.punish_interfer_PU

                        #self.reward[k] = self.reward[k]*0.9
                else:
                    #print("false")
                    success[k] = success[k] - self.senselength
                    success[k] = -1
                    # collision with PU##被干扰情况
                    m=sum(self.channel_state[i] for i in self.channel_position[action[k]])
                    self.fail_PU = self.fail_PU + (self.senselength-m)
                    rewards[k] = rewards[k]+self.punish_interfer_PU
                    # rewards[k] = rewards[k]+self.punish_interfer_PU
                    # if (len(np.where(action == action[k])[0]) > 1):
                    # if (not check_overlap_and_ones2(self.channel_position[action[k]], self.channel_state, self.channel_position, action)):
                        # collision with SU
                    #if (sum(self.channel_state[action[k]]) <self.Dk):
                    #if (sum(self.channel_state[i] for i in self.channel_position[action[k]])<3):
                    if (len(np.where(action == action[k])[0]) > 1):
                        self.fail_collision = self.fail_collision + 1
        # for k in range(self.num_agents):
        #     SU_sigma2[k][k] = 0
        # for k in range(self.num_agents):
        #     if (action[k] == self.n_channels):  # action is not choosing any channel
        #         rewards[k] = 0
        #     else:  # action is choosing one of channels
        #         for q in range(self.num_agents):
        #             if (action[q] == action[k]):
        #                 Interferecne_SU = Interferecne_SU + SU_sigma2[k][q] * self.SU_power
        #         SINR = self.H2[k, action[k]] * self.SU_power / (
        #                     Interferecne_SU + self.Interferecne_PU[k, action[k]] + self.Noise)
        #         rewards[k] = np.log2(1 + SINR)
        #         #mrewards[k] = np.log2(1 + SINR)
        #         if (self.channel_state[action[k]] == 1):
        #             if (len(np.where(action == action[k])[0]) == 1):
        #                 # successful transmission
        #                 self.success = self.success + 1
        #             else:
        #                 #rewards[k]=rewards[k]-1
        #                 # collision with SU
        #                 self.fail_collision = self.fail_collision + 1
        #         else:
        #             # collision with PU
        #             self.fail_PU = self.fail_PU + 1
        #             rewards[k] = self.punish_interfer_PU
        #             if (len(np.where(action == action[k])[0]) > 1):
        #                 # collision with SU
        #                 self.fail_collision = self.fail_collision + 1

        # Compute observation
        # obs = self.get_obs()#[array([0, 0, 0, 1, 1, 1]), array([0, 1, 0, 0, 0, 0]), array([1, 0, 0, 1, 1, 1])]
        #s_obs=

        # Compute done

        # Compute info
        info = {'agent_0': {}, 'agent_1': {}, 'agent_2': {}}
        done={'agent_0': False, 'agent_1': False, 'agent_2': False}
        obs = self.get_obs()#[array([0, 0, 0, 1, 1, 1]), array([0, 1, 0, 0, 0, 0]), array([1, 0, 0, 1, 1, 1])]

        #reward = [0, 0, 0]
        agents = ['agent_0', 'agent_1', 'agent_2']
        # rewards = defaultdict(int, zip(agents, rewards))
        ##rewards = defaultdict(int, zip(agents, success))
        rewards = defaultdict(int, zip(agents, rewards))

        #info = self.get_info()
        return obs, rewards, done, info


    def _action_to_list(self, a):
        if isinstance(a, np.ndarray):
            return a.tolist()
        if not isinstance(a, list):
            return [a]
        return a

    # def reset(self, seed=None, return_info=False, options=None):
    #
    #     self.agent_selection = self.env.agent_selection
    #
    #     self.rewards = np.zeros((self.num_agents,1))
    #     self.infos = {'agent_0': {}, 'agent_1': {}, 'agent_2': {}}
    #     self.done={'agent_0': False, 'agent_1': False, 'agent_2': False}
    #     self.agents = ['agent_0', 'agent_1', 'agent_2']##当前活动智能体列表，每次环境实例中，只有部分智能体是活动的
    #
    #     self._cumulative_rewards = np.zeros(self.num_agents,1)
    def render(self):
        # The probability of staying in current state in next time slot
        stay_prob = self.channel_state*self.stayGood_prob + (1-self.channel_state)*self.stayBad_prob
        tmp_dice = np.random.uniform(0, 1, self.n_channels+self.senselength-1) # roll the dice between 0 and 1
        stay_index = tmp_dice < stay_prob # 1: stay in current state, 0: change state

        # Update the channel state
        self.channel_state = self.channel_state*stay_index + (1-self.channel_state)*(1-stay_index)
        self.obs_block = [self.channel_state[i:i + self.senselength] for i in range(self.n_channels)]
        self.channel_position = [list(range(i, i + self.senselength)) for i in range(self.n_channels)]
        self.channelagg_state = [1 if sum(1 for bit in observation if bit == 1) >= self.Dk else 0 for observation in
                                 self.obs_block]

    def render_SINR(self):
        # Update the SINR

        # Calculate the channel gain
        SU_d = copy.deepcopy(np.reshape(self.SU_d, (-1, 1)))
        for n in range(self.n_channels+self.senselength-1-1):
            SU_d = np.hstack( (SU_d, np.reshape(self.SU_d, (-1, 1))) )
        less_than_8 = SU_d < 8
        SU_sigma2 = np.where(less_than_8, np.float_power(10, -((32.45 + 20 * np.log10(self.fc * SU_d)) / 10)),
                             np.float_power(10, -((58.3 + 33 * np.log10(SU_d / 8)) / 10)))
        #SU_sigma2=np.float_power(10, -((32.45+20*np.log10(self.fc*SU_d))/10))

        CN_real = np.random.normal(0, 1, size=(self.num_agents, self.n_channels+self.senselength-1))
        CN_imag = np.random.normal(0, 1, size=(self.num_agents, self.n_channels+self.senselength-1))
        theda = np.random.uniform(0, 1, size=(self.num_agents, self.n_channels+self.senselength-1))
        H = np.sqrt(self.K/(self.K+1)*SU_sigma2)*np.exp(1j*2*np.pi*theda) + np.sqrt(1/(self.K+1)*SU_sigma2/2)*(CN_real + 1j*CN_imag)

        self.H2 = np.float_power(np.absolute(H), 2)


        # Calculate the interference of PUs

        less_than_8_2 = self.SU_RX_PU_TX_d < 8
        PU_sigma2 = np.where(less_than_8_2,
                             np.float_power(10, -((32.45 + 20 * np.log10(self.fc * self.SU_RX_PU_TX_d)) / 10)),
                             np.float_power(10, -((58.3 + 33 * np.log10(self.SU_RX_PU_TX_d / 8)) / 10)))
        #PU_sigma2=np.float_power(10, -((32.45+20*np.log10(self.fc*self.SU_RX_PU_TX_d))/10))
        #PU_sigma2 = np.float_power(10, -((41 + 22.7 * np.log10(self.SU_RX_PU_TX_d) + 20 * np.log10(self.fc / 5)) / 10))
        channel_state = np.array([self.channel_state for k in range(self.num_agents)])

        self.Interferecne_PU = self.PU_power * PU_sigma2 * (1 - channel_state)


        self.SINR = self.H2*self.SU_power/(self.Interferecne_PU + self.Noise)
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


