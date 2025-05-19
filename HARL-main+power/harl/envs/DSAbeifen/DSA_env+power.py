import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import copy

from gym import spaces
random_seed=1
np.random.seed(random_seed)


class CRNEnvironment:
    def __init__(self, num_channels, noise_power, hU, hJ, M, SINR_threshold, rm, Cc, Cp):
        self.F = list(range(num_channels))  # 其中包含了频道的索引。这些频道索引表示了可用频道的标识。在代码中，self.F 的初始化使用了 range(num_channels)，它将创
        #channel子信道 or 子信道中心频率？
        # 建一个包含从 0 到 num_channels-1 的整数的列表。
        # 这个列表通常用于表示可供选择的频道，以便在频谱分配问题中选择要分配给用户的频道。通过创建包含频道索引的列表，代码可以轻松地在可用频道中进行选择和分配。
        self.P = [0.1, 0.5, 1.0]
        # 包含了不同的功率水平。这些功率水平表示不同的传输功率选项，可用于通信设备在频谱分配问题中选择。
        # 描述设备可以使用的不同传输功率选项。在频谱分配问题中，选择适当的传输功率水平是一个重要的决策因素，它可以影响通信质量、干扰水平以及电池寿命等方面。

        self.action_space = [(f, p) for f in self.F for p in self.P]  # 动作空间 action_space，其中包含了所有可能的动作组合。self.F 表示可用的频道（或通道）集合
        # self.P 表示可用的发射功率级别（或功率水平）集合使用列表推导式的方式，将频道和功率级别的组合构建成动作空间中的所有可能动作。
        # 需要获得action和observation，单智能体和多智能体不同
        # self.action_space = [(f, p) for f in self.F for p in self.P]#
        self.sinr_values = []  #

        self.num_channels = num_channels
        self.channels = np.arange(num_channels)
        self.noise_power = noise_power
        self.hU = hU
        self.hJ = hJ
        self.M = M
        self.SINR_threshold = SINR_threshold
        self.rm = rm
        self.Cc = Cc
        self.Cp = Cp
        self.spectrum_history = np.zeros((self.M, num_channels))
        self.previous_action = None

    # 接收端的信干噪比
    def calculate_SINR(self, PU_t, PJ_t, fU_t, fJ_t):
        interference = PJ_t * self.hJ * (fJ_t == fU_t)
        return PU_t * self.hU / (self.noise_power + interference)

    # 步骤函数，接收一个动作做完输入并且执行相应动作
    # 使用输入动作从动作空间获取实际动作
    # 从可用新的中随机选择一个信道
    def step(self, action):
        # 从动作空间获取实际动作
        actual_action = self.action_space[action]
        # 从动作空间获取实际的动作，包含可选动作的列表或者向量 action是一个索引
        fU_t, PU_t = actual_action
        # 从 actual_action 中解包出频道和传输功率分配的值

        # 干扰机频道&功率
        fJ_t = random.choice(self.channels)
        PJ_t = random.uniform(0, 1)
        #
        SINR_t = self.calculate_SINR(PU_t, PJ_t, fU_t, fJ_t)

        reward = SINR_t

        self.sinr_values.append(SINR_t)
        # 记录先前步骤智能体采取动作，参考它进行决策或者分析
        self.previous_action = actual_action
        # self.spectrum_history 在垂直方向上向上或向下滚动一个位置。numpy函数可以按照指定移动数目将数组元素沿指定轴滚动
        # 指定滚动的方向和距离
        self.spectrum_history = np.roll(self.spectrum_history, shift=-1, axis=0)
        #
        self.spectrum_history[-1, :] = np.random.randn(self.num_channels)
        # 判断是否低于阈值，作为触发条件用于是否需要采取进一步行动或决策
        return self.spectrum_history, reward, SINR_t < self.SINR_threshold

    # 重置环境状态，全零矩阵？None 环境重置为初始状态
    def reset(self):
        self.spectrum_history = np.zeros((self.M, self.num_channels))
        self.previous_action = None
        return self.spectrum_history


