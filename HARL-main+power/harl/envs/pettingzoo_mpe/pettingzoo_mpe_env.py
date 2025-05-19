import copy
import importlib
import logging
import numpy as np
import supersuit as ss

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


class PettingZooMPEEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = args["scenario"]##pettingzoo里面环境选择，我这边可以不设置
        del self.args["scenario"]
        self.discrete = True##动作离散
        if (
            "continuous_actions" in self.args
            and self.args["continuous_actions"] == True
        ):
            self.discrete = False
        #print(self.args)
        ##设置环境的最大周期数max_cycles，这个数字是在环境中随机生成的，但是我这边设置为25，也就是说环境中的周期数是25
        #当达到最大周期数时环境会自动重置;设置最大周期数原因：如果环境中的周期数设置为很大，会导致环境中的周期数过大，导致环境中的周期数过大，导致环境中的周期数过大，导致环境中的周期数过大，导致环境中的周期数过大，导致环境中的周期数过大，导致环境中的周期数过大，导致环境中的周期数过大，导致环境中的周期数过大，导致环境中的周期数过大，导致环�
        #print(self.max_cycles)
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 25
            self.args["max_cycles"] = 26
        self.cur_step = 0#设置当前步数初始值
        self.module = importlib.import_module("pettingzoo.mpe." + self.scenario)#导入pettingzoo特定环境
        #print("module", self.module)
        self.env = ss.pad_action_space_v0(
            ss.pad_observations_v0(self.module.parallel_env(**self.args))
        )##创建pettingzoo环境，对其进行一些预处理；这里是对环境的观测和动作空间进行了扩展，这个函数是supersuit里面的函数
        #print("env", self.env)

        self.env.reset()##重置环境
        self.n_agents = self.env.num_agents
        #print("n_agents", self.n_agents)#3
        self.agents = self.env.agents##当前活动智能体列表，每次环境实例中，只有部分智能体是活动的
        # print("agents", self.agents)##agents ['agent_0', 'agent_1', 'agent_2']
        # print(self.env.state_space)
        self.share_observation_space = self.repeat(self.env.state_space)
       # print("share_observation_space", self.share_observation_space)##share_observation_space [Box(-inf, inf, (54,), float32), Box(-inf, inf, (54,), float32), Box(-inf, inf, (54,), float32)]

        self.observation_space = self.unwrap(self.env.observation_spaces)
        # print("observation_space", self.observation_space)##observation_space [Box(-inf, inf, (18,), float32), Box(-inf, inf, (18,), float32), Box(-inf, inf, (18,), float32)]
        # print(self.env.action_spaces)
        self.action_space = self.unwrap(self.env.action_spaces)
        #包含三个Box对象，每个环境中有三个智能体，每个智能体都有自己动作空间
        # print("action_space", self.action_space)##action_space [Box(0.0, 1.0, (5,), float32), Box(0.0, 1.0, (5,), float32), Box(0.0, 1.0, (5,), float32)]
        self._seed = 0

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        # print(actions)#数组3行5列，因为选择了5步作为一组
        # print("act")
        #帮助控制每个回合的长度，在某些情况下如果没有设置最大步数，智能体可能会在异构回合中持续很长时间，可能会导致
        #学习过程变得困难。通过设置最大步数，可以确保每个回合在一定的步数后结束，从而使得学习过程更加稳定
        #max_cycles的值可以通过args参数设置，如果args没有提供max_cycles默认值会被设置25
        ##动作数组actions展平，调用wrap方法讲展平后动作数组转换为一个字典，字典键是智能体名称，值是对应智能体动作
        ##
        if self.discrete:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions.flatten()))
        else:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions))
        # print("actions")
        # print(actions)
        # print("obs")
        # print("Your output here", flush=True)
        # print(obs)
        # print("Your output here", flush=True)
        #
        # print("rew")
        # print(rew)
        # print(term)
        # print(trunc)
        # print(info)
        # print("info")
        # print("Your output here", flush=True)
       #  print(obs,"obsssssssss")#{'agent_0': array([ 0.28955808, -0.61648667, -0.33485594,  0.8138615 ,  0.20891371,
       # -1.0859568 ,  0.50903016, -1.6057136 , -0.23815146, -0.42803216,
       # -0.33272785, -1.2293934 , -0.7252434 , -1.8530444 ,  0.        ,
       #  0.        ,  0.        ,  0.        ], dtype=float32), 'agent_1': array([ 0.11473373,  0.12877865, -0.66758376, -0.4155319 ,  0.5416416 ,
       #  0.14343658,  0.841758  , -0.37632027,  0.09457639,  0.8013612 ,
       #  0.33272785,  1.2293934 , -0.39251554, -0.62365097,  0.        ,
       #  0.        ,  0.        ,  0.        ], dtype=float32), 'agent_2': array([-0.3823205 , -0.10766108, -1.0600994 , -1.0391829 ,  0.9341571 ,
       #  0.7670875 ,  1.2342736 ,  0.24733068,  0.48709193,  1.4250121 ,
       #  0.7252434 ,  1.8530444 ,  0.39251554,  0.62365097,  0.        ,
       #  0.        ,  0.        ,  0.        ], dtype=float32)

        self.cur_step += 1
        # print(term)
        # print("term")#
        # print("info")
        # print(info)
        #达到最大步数时结束当前回合，标记所有智能体转换为不良
        if self.cur_step == self.max_cycles:
            #创建字典，字典键是智能体，值是true，表示所有智能体需要被截断，即所有智能体需要结束当前回合
            trunc = {agent: True for agent in self.agents}
            #这行代码将每个智能体的 info 字典中的 "bad_transition" 键的值设置为 True。这表示这个智能体的转换是不良的，也就是说，这个智能体在当前回合的最后一步没有达到预期的状态。
            for agent in self.agents:
                info[agent]["bad_transition"] = True
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        ##定义state方法，应该返回环境的全局视图(环境的全局状态)；全局状态通常用于集中式训练和分散式执行的方法，比如QMIX
        #这个方法没有被实现，只是抛出异常错误
        ##54 all state
        #repeat()#each 54个 全局状态
        #print(self.env.state())##提供的state值是一个numpy数组，这个数组可能是从某个智能体视角得到的环境状态，或者是环境的某个局部状态
        s_obs = self.repeat(self.env.state())
        # print(("obs",obs,"s_obs",s_obs,"rew", rew, "term",term, trunc, info))
        # print("test111111111111")
        # print(s_obs)#{'agent_0': False, 'agent_1': False, 'agent_2': False}
        # print("s_obs")
        # print("test12222222222222222222222222222222222222222222222222222222")
        # print(rew)
        # for ag in self.agents:
        #     print(ag)
        #     print(rew[ag])
        # print("111111111111111111111111111111111111111111111111111111111111111111")
        # print("Your output here", flush=True)
        #print(rew,"rewww")#defaultdict(<class 'int'>, {'agent_0': -1.3565526895609565, 'agent_1': -1.3565526895609565, 'agent_2': -1.3565526895609565}) rewww
        total_reward = sum([rew[agent] for agent in self.agents])

        rewards = [[total_reward]] * self.n_agents
        # print(obs,"obs")
        # print(s_obs,"s_obs")
        # print(rewards,"rewards")
        #print(self.get_avail_actions(),"ava")#[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] ava

        # print(("rewards",rewards,"total",total_reward))
        # print("rewards")
        # print(rewards)
        # print("dones")
        # print(dones)
        # print("info")
        # print(info)
        # print("s_obs")
        # print(s_obs)
        # print("obs")
        # print(obs)
        # print(rew)#
        # print(rewards)##[[-3.781317302087892], [-3.781317302087892], [-3.781317302087892]]
        # print("sss")
        return (
            self.unwrap(obs),
            s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )
    ##强化学习环境中重置环境到初始状态，并返回初始的观察和状态
    ##
    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1##增加随机种子的值，随机种子用于控制环境的随机性，以便在需要的时候重现相同的环境状态
        self.cur_step = 0##这行代码将当前步数重置为0。这是因为每次开始新的回合时，都需要从第0步开始。
        obs = self.unwrap(self.env.reset(seed=self._seed))#
        s_obs = self.repeat(self.env.state())#
        return obs, s_obs, self.get_avail_actions()#
    ##离散动作空间和连续动作空间处理不同
    #对于离散动作空间，智能体选择的动作是有限的，可以通过get_avail_agent_actions 方法来获取所有可能的动作，这个方法返回一个列表，其中包含了所有可用动作
    ##对于连续动作空间，智能体可用选择的动作是无限的，不能通过列举所有可能的动作获取可用动作，这种情况下，智能体的动作通常由一个策略函数直接输出，这个策略函数可以是一个神经网络，接收当前的状态作为输入，然后输出一个连续的动作值
    ##
    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None
    #智能体的动作空间的大小，也就是智能体可以选择的动作的数量
    #这是列表，列表的长度等于智能体的动作空间的大小，列表中每个元素都是1，表示所有的动作都是可用的
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n#智能体的动作空间的大小，也就是智能体可用选择的动作的数量
    #列表长度等于智能体动作空间的大小，列表中的每个元素都是1，表示所有的动作都是可用的；所以这个方法是返回一个列表，这个列表表示给定智能体所有动作都是可用的

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
#接收一个字典d作为参数，这个字典的键是智能体的名称，值是对应智能体的某种信息(比如观察、奖励、是否完成等)

    def unwrap(self, d):
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
