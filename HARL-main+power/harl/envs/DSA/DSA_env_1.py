from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import copy
import random

from gym import spaces
random_seed=1
np.random.seed(random_seed)

class DSA_Markov():
    def __init__(
            self,
            nc_all,  #[25,25,25]
            n_channels,  #25 在所有信道中感知聚合频带感知了几次，也就是说所有信道中有多少个聚合频带
            num_agents,  #3
            sense_error_prob_max = 0.1,
            punish_interfer_PU = -5
    ):
        self.cur_step = 0#初始值
        self.nc_all = nc_all
        self.n_channels = n_channels #在所有信道中感知聚合频带感知了几次，也就是说所有信道中有多少个聚合频带
        self.senselength = 8 #感知长度
        self.DK = [2,4,6]  #带宽要求
        self.num_agents = num_agents # The number of the SUs

        self._has_reset = False
        self._has_rendered = False
        self._has_updated = False

        #初始化马尔可夫环境
        self._build_Markov_channel()
        #self.generate_Dk()
        #初始化位置
        self._build_location()

        self.Noise = 1 * np.float_power(10, -8)
        self.fc = 5
        self.K = 5
        self.SU_power = 20
        self.PU_power = 40

        self.render_SINR()

        self.n_actions = n_channels + 1 # The action space size 26
        self.n_features = n_channels # The sensing result space 25

        self.sense_error_prob_max = sense_error_prob_max #0.1
        self.sense_error_prob = np.random.uniform(0, self.sense_error_prob_max, size=(self.num_agents, self.n_channels+self.senselength-1))
        self.sense_error_probagent1 = np.random.uniform(0, self.sense_error_prob_max, size=(self.num_agents, self.nc_all[0]))
        #self.sense_error_prob_agg = np.random.uniform(0, self.sense_error_prob_max, size=(self.num_agents, self.n_channels-self.senselength+1))
        #self.n_channels+self.senselength-1 是所有信道的长度，一共有多少信道，信道L

        self.punish_interfer_PU = punish_interfer_PU
        self._seed = 1
        self.time_step = 0

        # 新增协议区域定义
        self.PU_CHANNELS = range(15, 32)  # 单独信道17-31
        self.ALOHA_CHANNELS = range(0, 11)  # 单独信道0-10
        self.TDMA_CHANNELS = range(11, 15)  # 单独信道11-16

        self.channel_position = [list(range(i, i + self.senselength)) for i in range(self.n_channels)]
        self.protocol_ratios = []  # 每个元素是[PU_ratio, ALOHA_ratio, TDMA_ratio]
        for cb in self.channel_position:
            pu = len([ch for ch in cb if ch in self.PU_CHANNELS])
            aloha = len([ch for ch in cb if ch in self.ALOHA_CHANNELS])
            tdma = len([ch for ch in cb if ch in self.TDMA_CHANNELS])
            total = len(cb)
            self.protocol_ratios.append([
                pu / total,
                aloha / total,
                tdma / total
            ])

    def _build_Markov_channel(self):
        self.channel_state = np.random.choice(2, self.n_channels+self.senselength-1)
        self.obs_block = [self.channel_state[i:i + self.senselength] for i in range(self.n_channels)]
        #存储的是每个信道的状态，每个信道的状态包含 senselength个元素
        self.channel_position = [list(range(i, i + self.senselength)) for i in range(self.n_channels)]
        #存储的是每个信道的频带索引位置，表示从哪个索引开始到哪个索引结束
        self.channelagg_states = [
            [1 if sum(obs) >= dk else 0 for obs in self.obs_block]
            for dk in self.DK  # DK应保持为[2,4,6]
        ]
        #判断当前聚合信道是否满足传输要求？8个信道中如果超过4个信道被占用，聚合信道状态为1，否则为0
        #observation只是一个中间变量

        self.stayGood_prob = np.random.uniform(0.7, 1, self.n_channels + self.senselength - 1)
        self.stayBad_prob = np.random.uniform(0, 0.3, self.n_channels + self.senselength - 1)
        self.goodToBad_prob = 1 - self.stayGood_prob
        self.badToGood_prob = 1 - self.stayBad_prob
        self.stayGood_prob_agg = np.random.uniform(0.7, 1, self.n_channels)
        self.stayBad_prob_agg = np.random.uniform(0, 0.3, self.n_channels)
        self.goodToBad_prob_agg = 1 - self.stayGood_prob_agg
        self.badToGood_prob_agg = 1 - self.stayBad_prob_agg

    def _build_location(self):

        # 初始化PU地址
        self.PU_TX_x = np.random.uniform(0, 200, self.n_channels + self.senselength - 1)
        self.PU_TX_y = np.random.uniform(0, 200, self.n_channels + self.senselength - 1)
        self.PU_RX_x = np.random.uniform(0, 200, self.n_channels + self.senselength - 1)
        self.PU_RX_y = np.random.uniform(0, 200, self.n_channels + self.senselength - 1)

        # 初始化SU发射器地址和SU接收器距离，SU_d是一个过渡变量
        self.SU_TX_x = np.random.uniform(0+40, 200-40, self.num_agents)
        self.SU_TX_y = np.random.uniform(0+40, 200-40, self.num_agents)
        self.SU_d = np.random.uniform(20, 100, self.num_agents)

        # 初始化SU接收器地址，SU_RX才是次级用户接收器地址
        SU_theda = 2 * np.pi * np.random.uniform(0, 1, self.num_agents)
        SU_dx = self.SU_d * np.cos(SU_theda)
        SU_dy = self.SU_d * np.sin(SU_theda)
        self.SU_RX_x = self.SU_TX_x + SU_dx
        self.SU_RX_y = self.SU_TX_y + SU_dy

        # PU发射器和SU接收器距离，主用户数量和总信道是一致，次级用户数量和代理数量是一致的
        self.SU_RX_PU_TX_d = np.zeros((self.num_agents, self.n_channels + self.senselength - 1))
        for k in range(self.num_agents):
            for l in range(self.n_channels + self.senselength - 1):
                self.SU_RX_PU_TX_d[k][l] = np.sqrt(
                    np.float_power(self.SU_RX_x[k] - self.PU_TX_x[l], 2) + np.float_power(
                        self.SU_RX_y[k] - self.PU_TX_y[l], 2))

        # SU发射器和SU接收器距离，二者都和代理数目一致
        self.SU_RX_SU_TX_d = np.zeros((self.num_agents, self.num_agents))
        for k1 in range(self.num_agents):
            for k2 in range(self.num_agents):
                self.SU_RX_SU_TX_d[k1][k2] = np.sqrt(
                    np.float_power(self.SU_RX_x[k1] - self.SU_TX_x[k2], 2) + np.float_power(
                        self.SU_RX_y[k1] - self.SU_TX_y[k2], 2))

    def close(self):
        """Closes the rendering window."""
        pass

    def store_action(self, action):
        self.action = action

    def sense(self):
        tmp_dice = np.random.uniform(0, 1, size=(self.num_agents, self.n_channels+self.senselength-1))  # roll the dice between 0 and 1
        error_index = tmp_dice < self.sense_error_prob # True: sensing error happens, False: sensing is correct
        self.sensing_result = self.channel_state*(1-error_index) + (1-self.channel_state)*(error_index)
        #sensing_result 通过布尔运算得出感知结果
        return self.sensing_result

    def get_state(self):
        self.getstate1=self.sense()
        self.getstate=self.getstate1[0]
        self.obs_block = [self.channel_state[i:i + self.senselength] for i in range(self.n_channels)]
        self.channel_position = [list(range(i, i + self.senselength)) for i in range(self.n_channels)]
        self.channelagg_states = [
            [1 if sum(obs) >= dk else 0 for obs in self.obs_block]
            for dk in self.DK  # DK应保持为[2,4,6]
        ]
        return self.getstate

    def get_obs(self,action):
        obs2_all=self.get_state() #这一步得到的是 sensing result[0]
        obsblock=[[obs2_all[i:i + self.senselength] for i in range(self.n_channels)],
                  [obs2_all[i:i + self.senselength] for i in range(self.n_channels)],
                  [obs2_all[i:i + self.senselength] for i in range(self.n_channels)]]
        # obs_block1 = [obs2_all[i:i + self.senselength] for i in range(self.n_channels)]
        # obs_block2 = [obs2_all[i:i + self.senselength] for i in range(self.n_channels)]
        # obs_block3 = [obs2_all[i:i + self.senselength] for i in range(self.n_channels)]
        obs2 = [[0] * self.senselength for _ in range(self.num_agents)]
        #obs2=[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]

        for i in range(self.num_agents):
            if action[i] == self.n_channels:
                obs2[i] = np.zeros(self.senselength)
            else:
                obs2[i] = obsblock[i][action[i]]
        agents = ['agent_0', 'agent_1', 'agent_2']
        obs = {agent: observation for agent, observation in zip(agents, obs2)}

        return obs

    def check_overlap_and_ones(self,pos1, pos2, channel_state, Dk):
        # pos1和pos2转换为集合并找出重叠部分
        overlap = set(pos1).intersection(set(pos2))
        # 找出不重叠部分的位置
        non_overlap_pos1 = [pos for pos in pos1 if pos not in overlap]
        non_overlap_pos2 = [pos for pos in pos2 if pos not in overlap]
        # 找出不重叠部分的数据
        non_overlap_data1 = [channel_state[pos] for pos in non_overlap_pos1]
        non_overlap_data2 = [channel_state[pos] for pos in non_overlap_pos2]
        # 如果不重叠部分数据为1的数量大于等于4
        if sum(non_overlap_data1) >= Dk or sum(non_overlap_data2) >= Dk or not overlap:
            return False  # False表示无冲突，可以继续选择这个频道
        else:
            return True  # True表示有冲突

    def check_overlap_and_ones2(self,pos2, channel_state, channel_blockposition, action, Dk):
        for i in range(len(action)):
            if (action[i] != self.n_channels):
                pos1 = channel_blockposition[action[i]]
                if pos1 != pos2:
                    overlap = set(pos1).intersection(set(pos2))
                    non_overlap_pos1 = [pos for pos in pos1 if pos not in overlap]
                    non_overlap_pos2 = [pos for pos in pos2 if pos not in overlap]
                    non_overlap_data1 = [channel_state[pos] for pos in non_overlap_pos1]
                    non_overlap_data2 = [channel_state[pos] for pos in non_overlap_pos2]
                    if sum(non_overlap_data2) < Dk:
                        return False, sum(non_overlap_data2)  # 返回False是不能传输
        return True, sum(pos1)  # 返回True是可以传输的意思


    def step(self, action, Dk):
        self.success = 0
        self.fail_collision = 0
        self.fail_ALOHA = 0
        self.fail_TDMA = 0
        self.fail_PU = 0

        self.render()
        self.render_SINR()

        rewards = np.zeros((self.num_agents))

        current_slot = self.time_step % 10  # TDMA周期为10个时隙
        tdma_active = (current_slot < 3)  # 前3个时隙激活

        SU_sigma2 = np.float_power(10, -((41 + 22.7 * np.log10(self.SU_RX_SU_TX_d) + 20 * np.log10(self.fc / 5)) / 10))
        #此处计算的是θ^2,而不是直接的 path loss, θ^2表示信号的强度大小，从而更真实地反映信号在直射路径和散射路径下的实际传输情况。
        for k in range(self.num_agents):
            SU_sigma2[k][k] = 0

        for k in range(self.num_agents):  # 0,1,2
            Interferecne_SU = 0  # 干扰
            if (action[k] == self.n_channels):#or self.channelagg_state[k]==0: # action is not choosing any channel
                rewards[k] = -3

            else: # action is choosing one of channel blocks
                # 获取当前聚合信道的协议比例
                pu_ratio, aloha_ratio, tdma_ratio = self.protocol_ratios[action[k]]
                dk_index = self.DK.index(Dk[k])  # 获取当前Dk对应的索引
                current_agg_state = self.channelagg_states[dk_index]  # 选择对应的状态列表

                for q in range(self.num_agents):
                    if action[q] != self.n_channels and self.check_overlap_and_ones(self.channel_position[action[k]], self.channel_position[action[q]],self.channel_state, Dk[k]):  # 表示代理 q选择了一个有效的频道
                        Interferecne_SU = Interferecne_SU + SU_sigma2[k][q] * self.SU_power  # 代理和代理的选择之间有冲突的话，SU的干扰则变成
                cb = self.channel_position[action[k]]  # cb等于代理 k选择的聚合信道每个信道的具体索引,action[k]表示代理 k(0,1,2)选择的聚合信道索引
                x_f = sum(self.H2[k, ch] * self.SU_power for ch in cb)
                y_f = sum(self.Interferecne_PU[k, ch] for ch in cb)
                SINR = x_f / (Interferecne_SU + y_f + self.Noise * Dk[k])
                base_reward = Dk[k] * np.log2(1 + SINR)

                aloha_collisions = 0
                if aloha_ratio >= 0:
                    # 统计选择信道块相邻且在ALOHA区域有显著重叠的冲突
                    same_cb_agents = [q for q in range(self.num_agents)
                                          if action[q] == action[k] and q != k]
                    aloha_collisions = len(same_cb_agents)

                # PU部分惩罚（原有逻辑）
                pu_penalty = self.punish_interfer_PU * pu_ratio if current_agg_state[action[k]] == 0 else 0

                # ALOHA部分奖励
                aloha_reward = 0
                if aloha_ratio >= 0:
                    aloha_reward = (5 * (aloha_collisions == 0) - 4 * aloha_collisions) * aloha_ratio

                # TDMA部分奖励
                tdma_reward = 0
                if tdma_ratio > 0:
                    if tdma_active:
                        tdma_reward = -4 * tdma_ratio
                    else:
                        tdma_reward = 4 * tdma_ratio

                # 组合奖励
                rewards[k] = base_reward + aloha_reward + tdma_reward + pu_penalty

                if current_agg_state[action[k]] == 1:
                    panduan, bili = self.check_overlap_and_ones2(self.channel_position[action[k]], self.channel_state, self.channel_position, action, Dk[k])
                    if (panduan):
                        #成功传输
                        self.success = self.success + 1   #怪怪的
                        rewards[k] = rewards[k]
                    else:
                        # 和SU碰撞失败
                        self.fail_collision = self.fail_collision+ 1
                        rewards[k] *= ((bili / self.num_agents) + (Dk[k] - bili)) / Dk[k]  # 这个公式很关键，轻易不要变动
                else:
                    # 和协议碰撞失败
                    rewards[k] = self.punish_interfer_PU
                    self.fail_PU = self.fail_PU + pu_ratio  # 当前步与主用户冲突失败次数
                    self.fail_ALOHA = self.fail_ALOHA + aloha_ratio
                    self.fail_TDMA = self.fail_TDMA + tdma_ratio
                    #print("fail_ALOHA:",self.fail_ALOHA)
                    #print("fail_TDMA:", self.fail_ALOHA)
                    if (len(np.where(action == action[k])[0]) > 1):
                        self.fail_collision = self.fail_collision + 1
                    # 如果多个代理选择了相同的信道块，则增加碰撞失败计数

        info = {'agent_0': {}, 'agent_1': {}, 'agent_2': {}}
        done={'agent_0': False, 'agent_1': False, 'agent_2': False}
        obs = self.get_obs(action)
        agents = ['agent_0', 'agent_1', 'agent_2']

        rewards = defaultdict(int, zip(agents, rewards))
        return obs, rewards, done, info


    def _action_to_list(self, a):
        if isinstance(a, np.ndarray):
            return a.tolist()
        if not isinstance(a, list):
            return [a]
        return a

    def render(self):
        # 定义协议区域（信道索引从0开始）
        TDMA_CYCLE = 10  # TDMA周期为10个时隙
        TDMA_ACTIVE_SLOTS = 3  # 每个周期前3个时隙为激活状态
        # 更新时间步（用于TDMA）
        self.time_step += 1
        # 遍历所有信道（假设总信道数为32）
        for ch in range(len(self.channel_state)):
            if ch in self.PU_CHANNELS:
                # ----------------------------------
                # PU区域（25-32）：马尔可夫链更新
                # ----------------------------------
                stay_prob = self.channel_state[ch] * self.stayGood_prob[ch] + (1 - self.channel_state[ch]) * \
                            self.stayBad_prob[ch]
                if np.random.rand() < stay_prob:
                    # 保持当前状态
                    pass
                else:
                    # 切换状态
                    self.channel_state[ch] = 1 - self.channel_state[ch]

            elif ch in self.ALOHA_CHANNELS:
                # ----------------------------------
                # ALOHA区域（0-10）：0.2概率接入
                # ----------------------------------
                if np.random.rand() < 0.2:
                    self.channel_state[ch] = 0  # 信道忙
                else:
                    self.channel_state[ch] = 1  # 信道空闲

            elif ch in self.TDMA_CHANNELS:
                # ----------------------------------
                # TDMA区域（9-16）：时隙轮转
                # ----------------------------------
                current_slot = self.time_step % TDMA_CYCLE
                if current_slot < TDMA_ACTIVE_SLOTS:
                    self.channel_state[ch] = 0  # 激活时隙
                else:
                    self.channel_state[ch] = 1  # 非激活时隙

            else:
                self.channel_state[ch] = 1
        #print("更新后的状态",self.channel_state)

        # 更新观测块和聚合信道状态（原有逻辑保留）
        self.obs_block = [self.channel_state[i:i + self.senselength] for i in range(self.n_channels)]
        self.channel_position = [list(range(i, i + self.senselength)) for i in range(self.n_channels)]
        self.channelagg_states = [
            [1 if sum(obs) >= dk else 0 for obs in self.obs_block]
            for dk in self.DK
        ]

    def render_SINR(self):
        # Update the SINR
        # Calculate the channel gain
        SU_d = copy.deepcopy(np.reshape(self.SU_d, (-1, 1)))
        for n in range(self.n_channels+self.senselength-1-1):
            SU_d = np.hstack((SU_d, np.reshape(self.SU_d, (-1, 1))))
        #less_than_8 = SU_d < 8
        SU_sigma2 = np.float_power(10, -((41 + 22.7 * np.log10(SU_d) + 20 * np.log10(self.fc / 5)) / 10))
        #SU_sigma2=np.float_power(10, -((32.45+20*np.log10(self.fc*SU_d))/10))

        CN_real = np.random.normal(0, 1, size=(self.num_agents, self.n_channels+self.senselength-1))
        CN_imag = np.random.normal(0, 1, size=(self.num_agents, self.n_channels+self.senselength-1))
        theda = np.random.uniform(0, 1, size=(self.num_agents, self.n_channels+self.senselength-1))
        H = np.sqrt(self.K/(self.K+1)*SU_sigma2)*np.exp(1j*2*np.pi*theda) + np.sqrt(1/(self.K+1)*SU_sigma2/2)*(CN_real + 1j*CN_imag)
        self.H2 = np.float_power(np.absolute(H), 2)
        #PU干扰

        PU_sigma2 = np.float_power(10, -((41 + 22.7 * np.log10(self.SU_RX_PU_TX_d) + 20 * np.log10(self.fc / 5)) / 10))

        channel_state = np.array([self.channel_state for k in range(self.num_agents)])

        self.Interferecne_PU = self.PU_power * PU_sigma2 * (1 - channel_state)

        self.SINR = self.H2 * self.SU_power/(self.Interferecne_PU + self.Noise)
        #信道功率反映了在特定条件下，信号从发送端到接收端的衰减情况，是评估信号质量的一个重要指标
        #SU_power 是指次级用户在发送信号时所使用的发射功率。这个功率值决定了次级用户的信号强度。
        #PU_power 是指主用户在其频谱上发送信号时所使用的发射功率。它决定了主用户信号的强度。


