"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
import copy

##将多个小图像拼接为一个大图像，例如在渲染多智能体环境的观察结果时
def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c

#使用cloudpickle库来序列化和反序列化内容，更强大的序列化工具，可以序列化任何python对象，包括函数和类不能被pickle库序列化的对象
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)

##是一个抽象基类，用于表示异步的、向量化的环境。这个类的主要目的是将多个环境的数据批量处理，使得每个观察变成一批观察，每个动作变成一批要在每个环境中执行的动作。
class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self, num_envs, observation_space, share_observation_space, action_space
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == "human":
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

##是一个在子进程中运行的函数，用于处理来自主进程的命令并与环境进行交互，接收三个参数：remote、parent_remote和env_fn_wrapper
##主要作用是在子进程中处理环境交互，从而实现环境的并行化
def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions,success_hist_1, fail_collision_hist_1, fail_PU_hist_1, fail_TDMA_hist_1, fail_ALOHA_hist_1= env.step(data)

            #ob, s_ob, reward, done, info, available_actions = env.step(data)
            if "bool" in done.__class__.__name__:  # done is a bool
                if (
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    info[0]["original_obs"] = copy.deepcopy(ob)
                    info[0]["original_state"] = copy.deepcopy(s_ob)
                    info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    info[0]["original_obs"] = copy.deepcopy(ob)
                    info[0]["original_state"] = copy.deepcopy(s_ob)
                    info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions, success_hist_1, fail_collision_hist_1, fail_PU_hist_1, fail_TDMA_hist_1, fail_ALOHA_hist_1))
        elif cmd == "reset":
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space)
            )
        elif cmd == "render_vulnerability":
            fr = env.render_vulnerability(data)
            remote.send((fr))
        elif cmd == "get_num_agents":
            remote.send((env.n_agents))
        else:
            raise NotImplementedError

##是sharevecenv的子类，用于创建并行环境，每个环境在一个子进程中运行，这个类主要目的是将多个环境的数据批量处理，使得每个观察变成一批观察，每个动作变成一批要在每个环境中执行的动作。
##实现环境的并行化，从而提高数据采样的效率
class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=shareworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_num_agents", None))
        self.n_agents = self.remotes[0].recv()
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[
            0
        ].recv()
        ShareVecEnv.__init__(
            self, len(env_fns), observation_space, share_observation_space, action_space
        )

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True
    #将动作 actions 发送到多个并行环境实例（self.remotes），并触发这些环境的步进操作（step）

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions, success_hist_1, fail_collision_hist_1, fail_PU_hist_1, fail_TDMA_hist_1, fail_ALOHA_hist_1 = zip(*results)
        #print((share_obs,"s"))
        #print((rews,"r"))

        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(rews),

            np.stack(dones),
            infos,
            np.stack(available_actions),
            np.stack(success_hist_1),
            np.stack(fail_collision_hist_1),
            np.stack(fail_PU_hist_1),
            np.stack(fail_TDMA_hist_1),
            np.stack(fail_ALOHA_hist_1),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        #print(obs,"9",share_obs,"9",available_actions)#18 54
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

##
# single env
class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None
        try:
            self.n_agents = env.n_agents
        except:
            pass

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions, success_hist_1, fail_collision_hist_1, fail_PU_hist_1, fail_TDMA_hist_1, fail_ALOHA_hist_1 = map(
            np.array, zip(*results)
        )

        for i, done in enumerate(dones):
            if "bool" in done.__class__.__name__:  # done is a bool
                if done:  # if done, save the original obs, state, and available actions in info, and then reset
                    infos[i][0]["original_obs"] = copy.deepcopy(obs[i])
                    infos[i][0]["original_state"] = copy.deepcopy(share_obs[i])
                    infos[i][0]["original_avail_actions"] = copy.deepcopy(available_actions[i])
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
                    success_hist_1[i], fail_collision_hist_1[i],fail_PU_hist_1[i],fail_TDMA_hist_1[i],fail_ALOHA_hist_1[i] = np.zeros_like(success_hist_1[i]), np.zeros_like(
                        fail_collision_hist_1[i]),np.zeros_like(fail_PU_hist_1[i]),np.zeros_like(fail_TDMA_hist_1[i]),np.zeros_like(fail_ALOHA_hist_1[i])
            else:
                if np.all(done):  # if done, save the original obs, state, and available actions in info, and then reset
                    infos[i][0]["original_obs"] = copy.deepcopy(obs[i])
                    infos[i][0]["original_state"] = copy.deepcopy(share_obs[i])
                    infos[i][0]["original_avail_actions"] = copy.deepcopy(available_actions[i])
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
                    success_hist_1[i], fail_collision_hist_1[i],fail_PU_hist_1[i],fail_TDMA_hist_1[i],fail_ALOHA_hist_1[i] = np.zeros_like(success_hist_1[i]), np.zeros_like(
                        fail_collision_hist_1[i]),np.zeros_like(fail_PU_hist_1[i]),np.zeros_like(fail_TDMA_hist_1[i]),np.zeros_like(fail_ALOHA_hist_1[i])

        self.actions = None

        return obs, share_obs, rews, dones, infos, available_actions, success_hist_1, fail_collision_hist_1, fail_PU_hist_1, fail_TDMA_hist_1, fail_ALOHA_hist_1

    # def step_wait(self):
    #     results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
    #     obs, share_obs, rews, dones, infos, available_actions,success_hist_1,fail_collision_hist_1= map(
    #         np.array, zip(*results)
    #     )
    #
    #     for i, done in enumerate(dones):
    #         if "bool" in done.__class__.__name__:  # done is a bool
    #             if (
    #                 done
    #             ):  # if done, save the original obs, state, and available actions in info, and then reset
    #                 infos[i][0]["original_obs"] = copy.deepcopy(obs[i])
    #                 infos[i][0]["original_state"] = copy.deepcopy(share_obs[i])
    #                 infos[i][0]["original_avail_actions"] = copy.deepcopy(
    #                     available_actions[i]
    #                 )
    #
    #                 obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
    #         else:
    #             if np.all(
    #                 done
    #             ):  # if done, save the original obs, state, and available actions in info, and then reset
    #                 infos[i][0]["original_obs"] = copy.deepcopy(obs[i])
    #                 infos[i][0]["original_state"] = copy.deepcopy(share_obs[i])
    #                 infos[i][0]["original_avail_actions"] = copy.deepcopy(
    #                     available_actions[i]
    #                 )
    #                 obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
    #     self.actions = None
    #
    #
    #     return obs, share_obs, rews, dones, infos, available_actions,success_hist_1,fail_collision_hist_1

    '''def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions'''

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = zip(*results)
        obs = np.array([np.array(o) for o in obs])
        share_obs = np.array([np.array(so) for so in share_obs])
        available_actions = np.array([np.array(aa) for aa in available_actions])
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError
