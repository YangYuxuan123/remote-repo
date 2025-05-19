"""Tools for HARL."""
import os
import random
import numpy as np
import torch
from harl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

#检查数组是否是一个numpy数组，如果是转换为pytorch张量，pytorch张量和numpy数组可以共享相同的内存，转换操作非常快
def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output

##change
def get_shape_from_obs_space(obs_space):
    # print("***************************")
    # print(obs_space)
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = (len(obs_space),)
        # obs_shape = obs_space
        #print("2")
    elif obs_space.__class__.__name__=='MultiBinary':
        obs_shape = (obs_space.n,)

    else:
        obs_shape = (len(obs_space),)
        #obs_shape=obs_space[0].shape # Outputs: (8,)
        #raise NotImplementedError
    #print(obs_shape)
    return obs_shape

#从动作空间act_space获取动作的形状
#act_space是智能体可以执行动作的空间
#函数返回值是动作形状
#离散动作空间：每个动作是一个整数，每个动作是一个整数
#MultiDiscrete 类型，那么动作的形状是 act_space.shape[0]。这是因为 MultiDiscrete 空间表示的是一个多元离散的动作空间，其中的每个动作都是一个整数向量，act_space.shape[0] 是这个向量的长度。
#Box 类型，那么动作的形状是 act_space.shape[0]。这是因为 Box 空间表示的是一个连续的动作空间，其中的每个动作都是一个实数向量，act_space.shape[0] 是这个向量的长度
# MultiBinary 类型，那么动作的形状是 act_space.shape[0]。这是因为 MultiBinary 空间表示的是一个多元二值的动作空间，其中的每个动作都是一个二值向量，act_space.shape[0] 是这个向量的长度
def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:
        act_shape=1
    return act_shape

##配置文件中设置训练参数：评估间隔 日志间隔 模型目录 并行环境数量 环境步数 训练间隔 每次训练更新次数 是否使用线性学习率衰减
##是否使用适当的时间限制 预热步数
def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    # if env_name == "dexhands":
    #     from harl.envs.dexhands.dexhands_env import DexHandsEnv
    #
    #     return DexHandsEnv({"n_threads": n_threads, **env_args})
    def get_env_fn(rank):
        def init_env():
            if env_name == "smac":
                from harl.envs.smac.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(env_args)
            elif env_name == "smacv2":
                from harl.envs.smacv2.smacv2_env import SMACv2Env

                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import (
                    MujocoMulti,
                )

                env = MujocoMulti(env_args=env_args)
            elif env_name == "pettingzoo_mpe":
                from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import (
                    PettingZooMPEEnv,
                )
                assert env_args["scenario"] in [
                    "simple_v2",
                    "simple_spread_v2",
                    "simple_reference_v2",
                    "simple_speaker_listener_v3",
                ], "only cooperative scenarios in MPE are supported"
                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from harl.envs.gym.gym_env import GYMEnv

                env = GYMEnv(env_args)
            elif env_name == "football":
                from harl.envs.football.football_env import FootballEnv

                env = FootballEnv(env_args)
            elif env_name == "lag":
                from harl.envs.lag.lag_env import LAGEnv

                env = LAGEnv(env_args)
            elif env_name == "DSA":
                from harl.envs.DSA.DSA_Markovenv import DSA_MarkovEnv
                env = DSA_MarkovEnv(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])

def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    if env_name == "dexhands":  # dexhands does not support running multiple instances
        raise NotImplementedError

    def get_env_fn(rank):
        def init_env():
            if env_name == "smac":
                from harl.envs.smac.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(env_args)
            elif env_name == "smacv2":
                from harl.envs.smacv2.smacv2_env import SMACv2Env

                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import (
                    MujocoMulti,
                )

                env = MujocoMulti(env_args=env_args)
            elif env_name == "pettingzoo_mpe":
                from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import (
                    PettingZooMPEEnv,
                )

                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from harl.envs.gym.gym_env import GYMEnv

                env = GYMEnv(env_args)
            elif env_name == "football":
                from harl.envs.football.football_env import FootballEnv

                env = FootballEnv(env_args)
            elif env_name == "lag":
                from harl.envs.lag.lag_env import LAGEnv

                env = LAGEnv(env_args)
            elif env_name == "DSA":
                from harl.envs.DSA.DSA_Markovenv import (DSA_MarkovEnv)

                env = DSA_MarkovEnv(env_args)

            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env
    #创建一个环境向量，这个环境向量可以并行的运行多个环境
    ##n_threads表示希望并行运行的环境的数量,在json文件都有；n_threads大于1则代表需要多个环境
    ##ShareSubprocVecEnv 类是一个环境向量，它在多个子进程中并行地运行环境。[get_env_fn(i) for i in range(n_threads)] 是一个列表推导式，它创建了一个列表，其中的每个元素都是一个函数，这个函数在被调用时会创建并初始化一个环境。
    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])

#函数作用是根据给定的环境名称和参数，创建一个用于渲染的环境并返回这个环境以及一些控制渲染行为的变量
def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    #变量用于控制渲染的行为；决定是否需要手动调用render函数；手动扩展并行环境的维度；决定是否需要收到延迟渲染
    manual_render = True  # manually call the render() function
    manual_expand_dims = True  # manually expand the num_of_parallel_envs dimension
    manual_delay = True  # manually delay the rendering by time.sleep()
    env_num = 1  # number of parallel envs
    if env_name == "smac":
        from harl.envs.smac.StarCraft2_Env import StarCraft2Env

        env = StarCraft2Env(args=env_args)
        manual_render = (
            False  # smac does not support manually calling the render() function
        )
        # instead, it use save_replay()
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "smacv2":
        from harl.envs.smacv2.smacv2_env import SMACv2Env

        env = SMACv2Env(args=env_args)
        manual_render = False
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "mamujoco":
        from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

        env = MujocoMulti(env_args=env_args)
        env.seed(seed * 60000)
    elif env_name == "pettingzoo_mpe":
        from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv

        env = PettingZooMPEEnv({**env_args, "render_mode": "human"})
        env.seed(seed * 60000)
    elif env_name == "gym":
        from harl.envs.gym.gym_env import GYMEnv

        env = GYMEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "football":
        from harl.envs.football.football_env import FootballEnv

        env = FootballEnv(env_args)
        manual_render = False  # football renders automatically
        env.seed(seed * 60000)
    elif env_name == "dexhands":
        from harl.envs.dexhands.dexhands_env import DexHandsEnv

        env = DexHandsEnv({"n_threads": 64, **env_args})
        manual_render = False  # dexhands renders automatically
        manual_expand_dims = (
            False  # dexhands uses parallel envs, thus dimension is already expanded
        )
        manual_delay = False
        env_num = 64
    elif env_name == "lag":
        from harl.envs.lag.lag_env import LAGEnv

        env = LAGEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "DSA":
        from harl.envs.DSA.DSA_Markovenv import DSA_MarkovEnv
        env = DSA_MarkovEnv(env_args)
        #manual_render = False  # football renders automatically
        env.seed(seed * 60000)
    else:
        print("Can not support the " + env_name + "environment.")
        raise NotImplementedError
    return env, manual_render, manual_expand_dims, manual_delay, env_num


def set_seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if env == "smac":
        from harl.envs.smac.smac_maps import get_map_params

        return get_map_params(env_args["map_name"])["n_agents"]
    elif env == "smacv2":
        return envs.n_agents
    elif env == "mamujoco":
        return envs.n_agents
    elif env == "pettingzoo_mpe":
        return envs.n_agents
    elif env == "gym":
        return envs.n_agents
    elif env == "football":
        return envs.n_agents
    elif env == "dexhands":
        return envs.n_agents
    elif env == "lag":
        return envs.n_agents
    elif env == "DSA":
        return envs.n_agents
