"""Base runner for off-policy algorithms."""
import os
import time
import torch
import numpy as np
import setproctitle
from harl.common.valuenorm import ValueNorm
from torch.distributions import Categorical
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config, get_task_name
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics import CRITIC_REGISTRY
from harl.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP
from harl.common.buffers.off_policy_buffer_fp import OffPolicyBufferFP


class OffPolicyBaseRunner:
    """Base runner for off-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OffPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        if "policy_freq" in self.algo_args["algo"]:
            self.policy_freq = self.algo_args["algo"]["policy_freq"]
        else:
            self.policy_freq = 1

        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.task_name = get_task_name(args["env"], env_args)
        if not self.algo_args["render"]["use_render"]:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)
            self.log_file = open(
                os.path.join(self.run_dir, "progress.txt"), "w", encoding="utf-8"
            )
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # env
        if self.algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        self.agent_deaths = np.zeros(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
        )

        self.action_spaces = self.envs.action_space
        # for agent_id in range(self.num_agents):
        #     self.action_spaces[agent_id].seed(algo_args["seed"]["seed"] + agent_id + 1)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        if self.share_param:
            self.actor = []
            agent = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            self.actor.append(agent)
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id]
                    == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                self.actor.append(agent)

        if not self.algo_args["render"]["use_render"]:
            self.critic = CRITIC_REGISTRY[args["algo"]](
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                self.envs.share_observation_space[0],
                self.envs.action_space,
                self.num_agents,
                self.state_type,
                device=self.device,
            )
            if self.state_type == "EP":
                self.buffer = OffPolicyBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    self.envs.share_observation_space[0],
                    self.num_agents,
                    self.envs.observation_space,
                    self.envs.action_space,
                )
            elif self.state_type == "FP":
                self.buffer = OffPolicyBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    self.envs.share_observation_space[0],
                    self.num_agents,
                    self.envs.observation_space,
                    self.envs.action_space,
                )
            else:
                raise NotImplementedError

        if (
            "use_valuenorm" in self.algo_args["train"].keys()
            and self.algo_args["train"]["use_valuenorm"]
        ):
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

        if self.algo_args["train"]["model_dir"] is not None:
            self.restore()

        self.total_it = 0  # total iteration

        if (
            "auto_alpha" in self.algo_args["algo"].keys()
            and self.algo_args["algo"]["auto_alpha"]
        ):
            self.target_entropy = []
            for agent_id in range(self.num_agents):
                if (
                    self.envs.action_space[agent_id].__class__.__name__ == "Box"
                ):  # Differential entropy can be negative
                    self.target_entropy.append(
                        -np.prod(self.envs.action_space[agent_id].shape)
                    )
                else:  # Discrete entropy is always positive. Thus we set the max possible entropy as the target entropy
                    self.target_entropy.append(
                        -0.98
                        * np.log(1.0 / np.prod(self.envs.action_space[agent_id].shape))
                    )
            self.log_alpha = []
            self.alpha_optimizer = []
            self.alpha = []
            for agent_id in range(self.num_agents):
                _log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.log_alpha.append(_log_alpha)
                self.alpha_optimizer.append(
                    torch.optim.Adam(
                        [_log_alpha], lr=self.algo_args["algo"]["alpha_lr"]
                    )
                )
                self.alpha.append(torch.exp(_log_alpha.detach()))
        elif "alpha" in self.algo_args["algo"].keys():
            self.alpha = [self.algo_args["algo"]["alpha"]] * self.num_agents

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["render"]["use_render"]:  # render, not train
            self.render()
            return
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        # warmup
        print("start warmup")
        obs, share_obs, available_actions = self.warmup()
        #print(obs,"obs")
        print("finish warmup, start training")
        # train and eval
        steps = (
            self.algo_args["train"]["num_env_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )
        update_num = int(  # update number per train
            self.algo_args["train"]["update_per_train"]
            * self.algo_args["train"]["train_interval"]
        )
        # obs = [torch.tensor(obs2) for obs2 in obs]
        # print(obs,"obs2")
        # available_actions = [torch.tensor(available_actions2) for available_actions2 in available_actions]
        for step in range(1, steps + 1):
            actions = self.get_actions(
                obs, available_actions=available_actions, add_random=True
            )
            #print(actions,"aomechaeng")
            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                new_available_actions,
            ) = self.envs.step(
                actions
            )  # rewards: (n_threads, n_agents, 1); dones: (n_threads, n_agents)
            #print(rewards,new_obs,"rewards")
            # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            next_available_actions = new_available_actions.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
            )
            #print(share_obs,"shar--")
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
            if step % self.algo_args["train"]["train_interval"] == 0:
                if self.algo_args["train"]["use_linear_lr_decay"]:
                    if self.share_param:
                        self.actor[0].lr_decay(step, steps)
                    else:
                        for agent_id in range(self.num_agents):
                            self.actor[agent_id].lr_decay(step, steps)
                    self.critic.lr_decay(step, steps)
                for _ in range(update_num):
                    self.train()
            if step % self.algo_args["train"]["eval_interval"] == 0:
                cur_step = (
                    self.algo_args["train"]["warmup_steps"]
                    + step * self.algo_args["train"]["n_rollout_threads"]
                )
                if self.algo_args["eval"]["use_eval"]:
                    print(
                        f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Evaluation at step {cur_step} / {self.algo_args['train']['num_env_steps']}:"
                    )
                    self.eval(cur_step)
                else:
                    print(
                        f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Step {cur_step} / {self.algo_args['train']['num_env_steps']}, average step reward in buffer: {self.buffer.get_mean_rewards()}.\n"
                    )
                    if len(self.done_episodes_rewards) > 0:
                        aver_episode_rewards = np.mean(self.done_episodes_rewards)
                        print(
                            "Some episodes done, average episode reward is {}.\n".format(
                                aver_episode_rewards
                            )
                        )
                        self.log_file.write(
                            ",".join(map(str, [cur_step, aver_episode_rewards])) + "\n"
                        )
                        self.log_file.flush()
                        self.done_episodes_rewards = []
                self.save()

    def warmup(self):
        """Warmup the replay buffer with random actions"""
        warmup_steps = (
            self.algo_args["train"]["warmup_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )
        # obs: (n_threads, n_agents, dim)
        # share_obs: (n_threads, n_agents, dim)
        # available_actions: (threads, n_agents, dim)
        obs, share_obs, available_actions = self.envs.reset()
        for _ in range(warmup_steps):
            # action: (n_threads, n_agents, dim)
            actions = self.sample_actions(available_actions)
            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                new_available_actions,
            ) = self.envs.step(actions)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            next_available_actions = new_available_actions.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
        return obs, share_obs, available_actions

    def insert(self, data):
        (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_agents, n_threads, obs_dim)
            actions,  # (n_agents, n_threads, action_dim)
            available_actions,  # None or (n_agents, n_threads, action_number)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: # list, shape: (n_threads, n_agents)
            next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
            next_obs,  # (n_threads, n_agents, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env

        # valid_transition denotes whether each transition is valid or not (invalid if corresponding agent is dead)
        # shape: (n_threads, n_agents, 1)
        valid_transitions = 1 - self.agent_deaths

        self.agent_deaths = np.expand_dims(dones, axis=-1)

        # terms use False to denote truncation and True to denote termination
        if self.state_type == "EP":
            terms = np.full((self.algo_args["train"]["n_rollout_threads"], 1), False)
            for i in range(self.algo_args["train"]["n_rollout_threads"]):
                if dones_env[i]:
                    if not (
                        "bad_transition" in infos[i][0].keys()
                        and infos[i][0]["bad_transition"] == True
                    ):
                        terms[i][0] = True
        elif self.state_type == "FP":
            terms = np.full(
                (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
                False,
            )
            for i in range(self.algo_args["train"]["n_rollout_threads"]):
                for agent_id in range(self.num_agents):
                    if dones[i][agent_id]:
                        if not (
                            "bad_transition" in infos[i][agent_id].keys()
                            and infos[i][agent_id]["bad_transition"] == True
                        ):
                            terms[i][agent_id][0] = True

        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[i]:
                self.done_episodes_rewards.append(self.train_episode_rewards[i])
                self.train_episode_rewards[i] = 0
                self.agent_deaths = np.zeros(
                    (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
                )
                if "original_obs" in infos[i][0]:
                    next_obs[i] = infos[i][0]["original_obs"].copy()
                if "original_state" in infos[i][0]:
                    next_share_obs[i] = infos[i][0]["original_state"].copy()

        if self.state_type == "EP":
            data = (
                share_obs[:, 0],  # (n_threads, share_obs_dim)
                obs,  # (n_agents, n_threads, obs_dim)
                actions,  # (n_agents, n_threads, action_dim)
                available_actions,  # None or (n_agents, n_threads, action_number)
                rewards[:, 0],  # (n_threads, 1)
                np.expand_dims(dones_env, axis=-1),  # (n_threads, 1)
                valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
                terms,  # (n_threads, 1)
                next_share_obs[:, 0],  # (n_threads, next_share_obs_dim)
                next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
                next_available_actions,  # None or (n_agents, n_threads, next_action_number)
            )
        elif self.state_type == "FP":
            data = (
                share_obs,  # (n_threads, n_agents, share_obs_dim)
                obs,  # (n_agents, n_threads, obs_dim)
                actions,  # (n_agents, n_threads, action_dim)
                available_actions,  # None or (n_agents, n_threads, action_number)
                rewards,  # (n_threads, n_agents, 1)
                np.expand_dims(dones, axis=-1),  # (n_threads, n_agents, 1)
                valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
                terms,  # (n_threads, n_agents, 1)
                next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
                next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
                next_available_actions,  # None or (n_agents, n_threads, next_action_number)
            )

        self.buffer.insert(data)

    def sample_actions(self, available_actions=None):
        """Sample random actions for warmup.
        Args:
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                 shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
        Returns:
            actions: (np.ndarray) sampled actions, shape is (n_threads, n_agents, dim)
        """
        actions = []
        for agent_id in range(self.num_agents):
            action = []
            for thread in range(self.algo_args["train"]["n_rollout_threads"]):
                if available_actions[thread] is None:
                    action.append(self.action_spaces[agent_id].sample())
                else:
                    action.append(
                        Categorical(
                            torch.tensor(available_actions[thread, agent_id, :])
                        ).sample()
                    )
            actions.append(action)
            #print(action,"mnnnn")

        #if self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
        return np.expand_dims(np.array(actions).transpose(1, 0), axis=-1)

        #return np.array(actions).transpose(1, 0, 2)

    @torch.no_grad()
    def get_actions(self, obs, available_actions=None, add_random=True):
        """Get actions for rollout.
        Args:
            obs: (np.ndarray) input observation, shape is (n_threads, n_agents, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                 shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
            add_random: (bool) whether to add randomness
        Returns:
            actions: (np.ndarray) agent actions, shape is (n_threads, n_agents, dim)
        """
        if self.args["algo"] == "hasac":
            actions = []
            for agent_id in range(self.num_agents):
                if (
                    len(np.array(available_actions).shape) == 3
                ):  # (n_threads, n_agents, action_number)
                    actions.append(
                        _t2n(
                            self.actor[agent_id].get_actions(
                                obs[:, agent_id],
                                available_actions[:, agent_id],
                                add_random,
                            )
                        )
                    )
                else:  # (n_threads, ) of None
                    actions.append(
                        _t2n(
                            self.actor[agent_id].get_actions(
                                obs[:, agent_id], stochastic=add_random
                            )
                        )
                    )
        else:
            actions = []
            for agent_id in range(self.num_agents):
                #print(obs[:, agent_id])
                actions.append(
                    _t2n(self.actor[agent_id].get_actions(obs[:, agent_id]))
                )
                # actions.append(
                #     _t2n(self.actor[agent_id].get_actions(obs[:, agent_id]))
                # )
            #print(actions,"trans")
            #actions = [action.reshape(-1, 1) for action in actions]

        return np.array(actions).transpose(1, 0, 2)

    def train(self):
        """Train the model"""
        raise NotImplementedError

    @torch.no_grad()
    def eval(self, step):
        """Evaluate the model"""
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
        eval_episode = 0
        if "smac" in self.args["env"]:
            eval_battles_won = 0
        if "football" in self.args["env"]:
            eval_score_cnt = 0
        episode_lens = []
        one_episode_len = np.zeros(
            self.algo_args["eval"]["n_eval_rollout_threads"], dtype=np.int_
        )

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        while True:
            eval_actions = self.get_actions(
                eval_obs, available_actions=eval_available_actions, add_random=False
            )
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            one_episode_len += 1

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    if "smac" in self.args["env"]:
                        if "v2" in self.args["env"]:
                            if eval_infos[eval_i][0]["battle_won"]:
                                eval_battles_won += 1
                        else:
                            if eval_infos[eval_i][0]["won"]:
                                eval_battles_won += 1
                    if "football" in self.args["env"]:
                        if eval_infos[eval_i][0]["score_reward"] > 0:
                            eval_score_cnt += 1
                    eval_episode_rewards[eval_i].append(
                        np.sum(one_episode_rewards[eval_i], axis=0)
                    )
                    one_episode_rewards[eval_i] = []
                    episode_lens.append(one_episode_len[eval_i].copy())
                    one_episode_len[eval_i] = 0

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                eval_episode_rewards = np.concatenate(
                    [rewards for rewards in eval_episode_rewards if rewards]
                )
                eval_avg_rew = np.mean(eval_episode_rewards)
                eval_avg_len = np.mean(episode_lens)
                if "smac" in self.args["env"]:
                    print(
                        "Eval win rate is {}, eval average episode rewards is {}, eval average episode length is {}.".format(
                            eval_battles_won / eval_episode, eval_avg_rew, eval_avg_len
                        )
                    )
                elif "football" in self.args["env"]:
                    print(
                        "Eval score rate is {}, eval average episode rewards is {}, eval average episode length is {}.".format(
                            eval_score_cnt / eval_episode, eval_avg_rew, eval_avg_len
                        )
                    )
                else:
                    print(
                        f"Eval average episode reward is {eval_avg_rew}, eval average episode length is {eval_avg_len}.\n"
                    )
                if "smac" in self.args["env"]:
                    self.log_file.write(
                        ",".join(
                            map(
                                str,
                                [
                                    step,
                                    eval_avg_rew,
                                    eval_avg_len,
                                    eval_battles_won / eval_episode,
                                ],
                            )
                        )
                        + "\n"
                    )
                elif "football" in self.args["env"]:
                    self.log_file.write(
                        ",".join(
                            map(
                                str,
                                [
                                    step,
                                    eval_avg_rew,
                                    eval_avg_len,
                                    eval_score_cnt / eval_episode,
                                ],
                            )
                        )
                        + "\n"
                    )
                else:
                    self.log_file.write(
                        ",".join(map(str, [step, eval_avg_rew, eval_avg_len])) + "\n"
                    )
                self.log_file.flush()
                self.writter.add_scalar(
                    "eval_average_episode_rewards", eval_avg_rew, step
                )
                self.writter.add_scalar(
                    "eval_average_episode_length", eval_avg_len, step
                )
                break

    @torch.no_grad()
    def render(self):
        """Render the model"""
        print("start rendering")
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = np.array([eval_available_actions])
                rewards = 0
                while True:
                    eval_actions = self.get_actions(
                        eval_obs,
                        available_actions=eval_available_actions,
                        add_random=False,
                    )
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = np.array([eval_available_actions])
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                rewards = 0
                while True:
                    eval_actions = self.get_actions(
                        eval_obs,
                        available_actions=eval_available_actions,
                        add_random=False,
                    )
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions)
                    rewards += eval_rewards[0][0][0]
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        if "smac" in self.args["env"]:  # replay for smac, no rendering
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def restore(self):
        """Restore the model"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].restore(self.algo_args["train"]["model_dir"], agent_id)
        if not self.algo_args["render"]["use_render"]:
            self.critic.restore(self.algo_args["train"]["model_dir"])
            if self.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/value_normalizer"
                    + ".pt"
                )
                self.value_normalizer.load_state_dict(value_normalizer_state_dict)

    def save(self):
        """Save the model"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].save(self.save_dir, agent_id)
        self.critic.save(self.save_dir)
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir) + "/value_normalizer" + ".pt",
            )

    def close(self):
        """Close environment, writter, and log file."""
        # post process
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.log_file.close()

# """Base runner for off-policy algorithms."""
# import os
# import time
# import torch
# import numpy as np
# import setproctitle
# from harl.common.valuenorm import ValueNorm
# from torch.distributions import Categorical
# from harl.utils.trans_tools import _t2n
# from harl.utils.envs_tools import (
#     make_eval_env,
#     make_train_env,
#     make_render_env,
#     set_seed,
#     get_num_agents,
# )
# from harl.utils.models_tools import init_device
# from harl.utils.configs_tools import init_dir, save_config, get_task_name
# from harl.algorithms.actors import ALGO_REGISTRY
# from harl.algorithms.critics import CRITIC_REGISTRY
# from harl.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP
# #这个类用于处理以 "Episode-Pruned"（EP）状态表示的环境。在这种情况下，每个智能体在每个时间步都有自己的观察，但所有智能体共享全局状态。这意味着，尽管每个智能体可能只能看到局部的信息，但它们都能访问到全局的信息。
# from harl.common.buffers.off_policy_buffer_fp import OffPolicyBufferFP
# #这个类用于处理以 "Feature-Pruned"（FP）状态表示的环境。在这种情况下，每个智能体在每个时间步都有自己的观察，并且全局状态对于每个智能体都可能不同。这意味着，每个智能体的全局状态都是根据其自身的观察和其他智能体的一些特征来构造的。
#
# class OffPolicyBaseRunner:
#     """Base runner for off-policy algorithms."""
#
#     def __init__(self, args, algo_args, env_args):
#         """Initialize the OffPolicyBaseRunner class.
#         Args:
#             args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
#             algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
#             env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
#         """
#         self.args = args#args命名空间;初始化类的实例
#         self.algo_args = algo_args#algo_args命名空间
#         self.env_args = env_args#env_args命名空间
#         #设置policy_freq参数；这个参数决定策略更新的频率。在每次训练迭代中，我们会多次更新策略，但只有在每隔 policy_freq 步时才会更新目标策略。
#         #参数通常用于决定在训练过程中，每多少步进行一次策略更新；policy-freq是1，那么每一步都会进行策略更新；是10，每十步进行一次策略更新
#         if "policy_freq" in self.algo_args["algo"]:
#             self.policy_freq = self.algo_args["algo"]["policy_freq"]
#         else:
#             self.policy_freq = 1
#         #参数值常用于指定环境的状态表示类型；例如EP可能表示 "Episode-Pruned" 状态表示
#         self.state_type = env_args.get("state_type", "EP")
#         self.share_param = algo_args["algo"]["share_param"]#。share_param 参数通常用于指定是否在智能体之间共享模型参数。
#         self.fixed_order = algo_args["algo"]["fixed_order"]#fixed_order 参数通常用于指定智能体的顺序是否固定。如果 fixed_order 为 True，则智能体的顺序是固定的，否则智能体的顺序是随机的。
#
#         #初始化实类变量
#         #
#         set_seed(algo_args["seed"])#设置随机数种子确保实验的可重复性
#         self.device = init_device(algo_args["device"])#调用设备，用以存储和计算张量
#         self.task_name = get_task_name(args["env"], env_args)#任务名称将被用于日志和模型保存
#         #初始化一些日志与配置相关的类实例变量
#         #use_render 参数通常用于指定是否在训练过程中进行渲染
#         #这些目录和写入器将用于保存训练过程中的日志和模型。
#         #
#         if not self.algo_args["render"]["use_render"]:
#             self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
#                 args["env"],
#                 env_args,
#                 args["algo"],
#                 args["exp_name"],
#                 algo_args["seed"]["seed"],
#                 logger_path=algo_args["logger"]["log_dir"],
#             )
#             save_config(args, algo_args, env_args, self.run_dir)#打开异构文件用于记录训练过程中的进度，这个文件位于运行目录下；训练过程中是否被写入
#             self.log_file = open(
#                 os.path.join(self.run_dir, "progress.txt"), "w", encoding="utf-8"
#             )
#         #setproctitle 库的一个函数，用于设置当前进程的标题。这个标题会显示在系统的进程列表中，例如在运行 top 或 ps 命令时。
#         #
#         setproctitle.setproctitle(
#             str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
#         )
#
#         # env
#         #创建用于渲染的环境，可视化智能体在环境中的行为，对于调解和调试智能体策略非常有用
#         if self.algo_args["render"]["use_render"]:  # make envs for rendering
#             (
#                 self.envs,
#                 self.manual_render,
#                 self.manual_expand_dims,
#                 self.manual_delay,
#                 self.env_num,
#             ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
#         #用于创建训练和评估环境
#         else:  # make envs for training and evaluation
#             #函数返回一个环境对象，这个环境1将被用于训练智能体
#             self.envs = make_train_env(
#                 args["env"],
#                 algo_args["seed"]["seed"],
#                 algo_args["train"]["n_rollout_threads"],
#                 env_args,
#             )
#             #这个函数返回一个环境对象，这个环境将被用于评估智能体的性能。如果 algo_args["eval"]["use_eval"] 的值为 False，那么 self.eval_envs 将被设置为 None。
#             self.eval_envs = (
#                 make_eval_env(
#                     args["env"],
#                     algo_args["seed"]["seed"],
#                     algo_args["eval"]["n_eval_rollout_threads"],
#                     env_args,
#                 )
#                 if algo_args["eval"]["use_eval"]#如果 algo_args["eval"]["use_eval"] 的值为 False，那么 self.eval_envs 将被设置为 None。
#                 else None
#             )
#         ##这段代码主要用于初始化智能体的数量，智能体的死亡状态以及动作空间
#         ##函数返回智能体数量，这个数量赋值为num_agents
#         self.num_agents = get_num_agents(args["env"], env_args, self.envs)
#         #这行代码创建一个全零的 NumPy 数组，用于记录每个智能体的死亡状态。这个数组的形状是 (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1)，其中 self.algo_args["train"]["n_rollout_threads"] 是并行环境的数量，self.num_agents 是智能体的数量。
#         self.agent_deaths = np.zeros(
#             (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
#         )
#         #print(self.algo_args["train"]["n_rollout_threads"],"roll")
#         ##这行代码获取环境的动作空间，并将其赋值给 self.action_spaces。
#         # 动作空间是一个列表，其中每个元素是一个动作空间对象，表示一个智能体可以执行的所有动作。
#         self.action_spaces = self.envs.action_space
#         ##对于每个智能体，这行代码设置其动作空间的随机种子。这个种子的值是 algo_args["seed"]["seed"] + agent_id + 1。设置随机种子可以确保实验的可重复性，即每次运行相同的代码都会得到相同的结果。
#         ##change 已经是随机生成不需要重新生成
#         # for agent_id in range(self.num_agents):
#         #     self.action_spaces[agent_id]=self.envs.action_space[agent_id]
#             #self.action_spaces[agent_id].seed(algo_args["seed"]["seed"] + agent_id + 1)
#         ##
#         print("share_observation_space: ", self.envs.share_observation_space)
#         print("observation_space: ", self.envs.observation_space)
#         print("action_space: ", self.envs.action_space)
#
#         if self.share_param:
#             self.actor = []#代码目的是创建智能体的策略网络(Actor),首先检查share_param的值；这个值决定是否在所有智能体之间共享模型参数
#             #如果 self.share_param 为 True，那么所有的智能体将共享同一个策略网络。首先，它创建一个策略网络实例，并将其添加到 self.actor 列表中。然后，对于剩余的每个智能体，它都会将这个策略网络实例添加到 self.actor 列表中。这意味着 self.actor 列表中的所有元素都是同一个策略网络实例。
#             #如果 self.share_param 为 False，那么每个智能体都将有自己独立的策略网络。它会为每个智能体创建一个新的策略网络实例，并将其添加到 self.actor 列表中。
#             #在创建策略网络实例时，它使用 ALGO_REGISTRY[args["algo"]] 来获取策略网络的类，然后调用这个类的构造函数来创建实例。这个构造函数接收四个参数：一个包含模型和算法参数的字典，观察空间，动作空间，以及设备。
#             #根据需求(是否共享模型参数)创建适当策略网络
#             agent = ALGO_REGISTRY[args["algo"]](
#                 {**algo_args["model"], **algo_args["algo"]},
#                 self.envs.observation_space[0],
#                 self.envs.action_space[0],
#                 device=self.device,
#             )
#             ##这行代码将 agent 对象添加到 self.actor 列表中。agent 是一个策略网络（Actor）的实例，它是通过 ALGO_REGISTRY[args["algo"]] 类的构造函数创建的。这个类是从 ALGO_REGISTRY 注册表中获取的，其中 args["algo"] 是你选择的算法的名称。ALGO_REGISTRY 是一个字典，将算法名称映射到相应的策略网络类。
#             #用于存储所有智能体的策略网络，在这个代码中首先添加第一个智能体的策略网络；在接下来循环中将为剩余的每个智能体添加策略网络
#             self.actor.append(agent)
#             ##当 self.share_param 为 True 时执行的。这意味着所有的智能体将共享同一个策略网络。
#             #这行代码遍历除第一个智能体外的所有智能体。
#             for agent_id in range(1, self.num_agents):
#                 ##assert这行代码检查当前智能体观察空间是否与第一个智能体观察空间相同；如果不同它将抛出一个断言错误，在观察空间不同的情况下，共享模型参数是无效的
#                 assert (
#                     self.envs.observation_space[agent_id]
#                     == self.envs.observation_space[0]
#                 ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
#                 ##这行代码检查当前智能体的动作空间是否与第一个智能体的动作空间相同。如果不同，那么它将抛出一个断言错误，因为在动作空间不同的情况下，共享模型参数是无效的。动作空间定义了智能体可以执行的所有动作的集合，如果两个智能体的动作空间不同，那么它们可能需要学习不同的策略来最大化其累积奖励。
#                 assert (
#                     self.envs.action_space[agent_id] == self.envs.action_space[0]
#                 ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
#                 #这行代码将第一个智能体的策略网络实例添加到 self.actor 列表中。这意味着 self.actor 列表中的所有元素都是同一个策略网络实例。这样，所有的智能体都将共享同一个策略网络，这可以减少模型的复杂性和计算需求，但前提是所有智能体的观察空间和动作空间必须相同。
#                 self.actor.append(self.actor[0])
#         #不共享模型参数；意味着每个智能体都将有自己独立的策略网络
#         else:
#             self.actor = []#用于存储所有智能体的策略网络
#             #这个构造函数接收四个参数：一个包含模型和算法参数的字典，观察空间，动作空间，以及设备。
#             for agent_id in range(self.num_agents):
#                 agent = ALGO_REGISTRY[args["algo"]](
#                     {**algo_args["model"], **algo_args["algo"]},
#                     self.envs.observation_space[agent_id],
#                     self.envs.action_space[agent_id],
#                     device=self.device,
#                 )
#                 self.actor.append(agent)
#         ##主要用于创建评论家网络(Critic)和经验回放缓冲区(Buffer)
#         ##是一个布尔值，通常用于指定是否在训练过程中进行环境渲染。如果其值为 False，则执行后续的代码块。这通常意味着在训练过程中不进行环境渲染，而是进行其他操作，如模型训练或评估。如果其值为 True，则跳过后续的代码块，可能在其他地方进行环境渲染。
#         if not self.algo_args["render"]["use_render"]:
#             #这行代码创建一个评论家网络实例。它首先从 CRITIC_REGISTRY 注册表中获取评论家网络的类，其中 args["algo"] 是你选择的算法的名称。然后，它调用这个类的构造函数来创建实例。这个构造函数接收六个参数：一个包含训练、模型和算法参数的字典，共享的观察空间，动作空间，智能体的数量，状态类型，以及设备。
#             self.critic = CRITIC_REGISTRY[args["algo"]](
#                 {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
#                 self.envs.share_observation_space[0],
#                 self.envs.action_space,
#                 self.num_agents,
#                 self.state_type,
#                 device=self.device,
#             )
#             ##这行代码检查 self.state_type 的值。如果为 "EP"，则执行以下代码块，用于创建 OffPolicyBufferEP 类型的经验回放缓冲区。
#             #
#             if self.state_type == "EP":
#                 self.buffer = OffPolicyBufferEP(
#                     {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
#                     self.envs.share_observation_space[0],
#                     self.num_agents,
#                     self.envs.observation_space,
#                     self.envs.action_space,
#                 )
#             elif self.state_type == "FP":
#                 self.buffer = OffPolicyBufferFP(
#                     {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
#                     self.envs.share_observation_space[0],
#                     self.num_agents,
#                     self.envs.observation_space,
#                     self.envs.action_space,
#                 )
#             else:
#                 raise NotImplementedError
#         ##
#         ##这行代码首先检查 self.algo_args["train"] 字典中是否包含 "use_valuenorm" 键，然后检查 "use_valuenorm" 的值是否为 True。如果这两个条件都满足，那么它将执行以下代码块，用于创建值归一化器。
#         if (
#             "use_valuenorm" in self.algo_args["train"].keys()
#             and self.algo_args["train"]["use_valuenorm"]
#         ):
#             self.value_normalizer = ValueNorm(1, device=self.device)#这行代码创建一个 ValueNorm 对象，并将其赋值给 self.value_normalizer。ValueNorm 是一个用于归一化值的类，它可以使值的分布接近标准正态分布。这个类的构造函数接收两个参数：值的维度（在这里是 1）和设备（self.device）。
#         #如果 self.algo_args["train"] 字典中不包含 "use_valuenorm" 键，或者 "use_valuenorm" 的值为 False，那么 self.value_normalizer 将被设置为 None。
#         else:
#             self.value_normalizer = None
#         ##这行代码检查 self.algo_args["train"]["model_dir"] 的值是否为 None。如果不是 None，那么它将调用 self.restore 方法来恢复模型。这个方法用于从指定的目录中加载模型的参数。这可以用于继续之前的训练，或者在已经训练好的模型上进行评估。
#         #根据需求和环境条件初始化值归一化器和恢复模型
#         if self.algo_args["train"]["model_dir"] is not None:
#             self.restore()
#         ##初始化次数；记录训练过程中的总迭代次数；每进行一次训练迭代，total_it 将增加 1。这个变量可以用于跟踪训练的进度；保存模型，调整学习率
#         self.total_it = 0  # total iteration
#         ##初始化alpha参数，这个参数在某些强化详细算法当中控制策略的熵，从而影响探索和利用的平衡；
#         #这行代码首先检查 self.algo_args["algo"] 字典中是否包含 "auto_alpha" 键，然后检查 "auto_alpha" 的值是否为 True。如果这两个条件都满足，那么它将执行以下代码块，用于自动调整 alpha 参数。
#         if (
#             "auto_alpha" in self.algo_args["algo"].keys()
#             and self.algo_args["algo"]["auto_alpha"]
#         ):
#             self.target_entropy = []##这行代码创建一个空列表 self.target_entropy，用于存储每个智能体的目标熵。
#             for agent_id in range(self.num_agents):
#                 #这行代码检查当前智能体的动作空间是否是连续的（即 Box 类型）。如果是，那么它将计算动作空间的负维度作为目标熵。否则，它将计算动作空间的最大可能熵作为目标熵。
#                 if (
#                     self.envs.action_space[agent_id].__class__.__name__ == "Box"
#                 ):  # Differential entropy can be negative
#                     ##
#                     self.target_entropy.append(
#                         -np.prod(self.envs.action_space[agent_id].shape)
#                     )
#                 else:  # Discrete entropy is always positive. Thus we set the max possible entropy as the target entropy
#                     self.target_entropy.append(
#                         -0.98
#                         * np.log(1.0 / np.prod(self.envs.action_space[agent_id].shape))
#                     )
#             self.log_alpha = []
#             self.alpha_optimizer = []
#             self.alpha = []
#             for agent_id in range(self.num_agents):
#                 _log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
#                 self.log_alpha.append(_log_alpha)
#                 self.alpha_optimizer.append(
#                     torch.optim.Adam(
#                         [_log_alpha], lr=self.algo_args["algo"]["alpha_lr"]
#                     )
#                 )
#                 self.alpha.append(torch.exp(_log_alpha.detach()))
#         ##这行代码创建一个列表 self.alpha，其中每个元素都是 self.algo_args["algo"]["alpha"]。这个列表的长度等于智能体的数量。
#         elif "alpha" in self.algo_args["algo"].keys():
#             self.alpha = [self.algo_args["algo"]["alpha"]] * self.num_agents
#
#     def run(self):
#         """Run the training (or rendering) pipeline."""
#         if self.algo_args["render"]["use_render"]:  # render, not train
#             self.render()
#             return
#         self.train_episode_rewards = np.zeros(
#             self.algo_args["train"]["n_rollout_threads"]
#         )
#         self.done_episodes_rewards = []
#         # warmup
#         print("start warmup")
#         obs, share_obs, available_actions = self.warmup()
#         print("finish warmup, start training")
#         # train and eval
#         steps = (
#             self.algo_args["train"]["num_env_steps"]
#             // self.algo_args["train"]["n_rollout_threads"]
#         )
#         update_num = int(  # update number per train
#             self.algo_args["train"]["update_per_train"]
#             * self.algo_args["train"]["train_interval"]
#         )
#         for step in range(1, steps + 1):
#             actions = self.get_actions(
#                 obs, available_actions=available_actions, add_random=True
#             )
#             #print(actions,"test89")
#             (
#                 new_obs,
#                 new_share_obs,
#                 rewards,
#                 dones,
#                 infos,
#                 new_available_actions,
#             ) = self.envs.step(
#                 actions
#             )
#             #print(rewards,obs,"tett")
#             # rewards: (n_threads, n_agents, 1); dones: (n_threads, n_agents)
#             # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
#             next_obs = new_obs.copy()
#             next_share_obs = new_share_obs.copy()
#             next_available_actions = new_available_actions.copy()
#             data = (
#                 share_obs,
#                 obs.transpose(1, 0, 2),
#                 actions.transpose(1, 0, 2),
#                 available_actions.transpose(1, 0, 2)
#                 if len(np.array(available_actions).shape) == 3
#                 else None,
#                 rewards,
#                 dones,
#                 infos,
#                 next_share_obs,
#                 next_obs,
#                 next_available_actions.transpose(1, 0, 2)
#                 if len(np.array(available_actions).shape) == 3
#                 else None,
#             )
#             #print(actions,"accccc")
#             #print(share_obs,"shar")
#             self.insert(data)
#             obs = new_obs
#             share_obs = new_share_obs
#             available_actions = new_available_actions
#             if step % self.algo_args["train"]["train_interval"] == 0:
#                 if self.algo_args["train"]["use_linear_lr_decay"]:
#                     if self.share_param:
#                         self.actor[0].lr_decay(step, steps)
#                     else:
#                         for agent_id in range(self.num_agents):
#                             self.actor[agent_id].lr_decay(step, steps)
#                     self.critic.lr_decay(step, steps)
#                 for _ in range(update_num):
#                     self.train()
#             if step % self.algo_args["train"]["eval_interval"] == 0:
#                 cur_step = (
#                     self.algo_args["train"]["warmup_steps"]
#                     + step * self.algo_args["train"]["n_rollout_threads"]
#                 )
#                 if self.algo_args["eval"]["use_eval"]:
#                     print(
#                         f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Evaluation at step {cur_step} / {self.algo_args['train']['num_env_steps']}:"
#                     )
#                     self.eval(cur_step)
#                 else:
#                     print(
#                         f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Step {cur_step} / {self.algo_args['train']['num_env_steps']}, average step reward in buffer: {self.buffer.get_mean_rewards()}.\n"
#                     )
#                     if len(self.done_episodes_rewards) > 0:
#                         aver_episode_rewards = np.mean(self.done_episodes_rewards)
#                         print(
#                             "Some episodes done, average episode reward is {}.\n".format(
#                                 aver_episode_rewards
#                             )
#                         )
#                         self.log_file.write(
#                             ",".join(map(str, [cur_step, aver_episode_rewards])) + "\n"
#                         )
#                         self.log_file.flush()
#                         self.done_episodes_rewards = []
#                 self.save()
#     def warmup(self):
#         """Warmup the replay buffer with random actions"""
#         warmup_steps = (
#             self.algo_args["train"]["warmup_steps"]
#             // self.algo_args["train"]["n_rollout_threads"]
#         )
#         # obs: (n_threads, n_agents, dim)
#         # share_obs: (n_threads, n_agents, dim)
#         # available_actions: (threads, n_agents, dim)
#         obs, share_obs, available_actions = self.envs.reset()
#         for _ in range(warmup_steps):
#             # action: (n_threads, n_agents, dim)
#             actions = self.sample_actions(available_actions)
#             (
#                 new_obs,
#                 new_share_obs,
#                 rewards,
#                 dones,
#                 infos,
#                 new_available_actions,
#             ) = self.envs.step(actions)
#             #print(actions,"testaaaaaaa")
#             next_obs = new_obs.copy()
#             next_share_obs = new_share_obs.copy()
#             next_available_actions = new_available_actions.copy()
#             data = (
#                 share_obs,
#                 obs.transpose(1, 0, 2),
#                 actions.transpose(1, 0, 2),
#                 available_actions.transpose(1, 0, 2)
#                 if len(np.array(available_actions).shape) == 3
#                 else None,
#                 rewards,
#                 dones,
#                 infos,
#                 next_share_obs,
#                 next_obs,
#                 next_available_actions.transpose(1, 0, 2)
#                 if len(np.array(available_actions).shape) == 3
#                 else None,
#             )
#             self.insert(data)
#             obs = new_obs
#             share_obs = new_share_obs
#             available_actions = new_available_actions
#         return obs, share_obs, available_actions
#
#     # def warmup(self):
#     #     """Warmup the replay buffer with random actions"""
#     #     ##每个并行环境需要进行的预热步骤数
#     #     warmup_steps = (
#     #         self.algo_args["train"]["warmup_steps"]
#     #         // self.algo_args["train"]["n_rollout_threads"]
#     #     )
#     #     # obs: (n_threads, n_agents, dim)
#     #     # share_obs: (n_threads, n_agents, dim)
#     #     # available_actions: (threads, n_agents, dim)
#     #
#     #     obs, share_obs, available_actions = self.envs.reset()
#     #     #print(obs,share_obs,available_actions,"test")
#     #
#     #     for _ in range(warmup_steps):
#     #         # action: (n_threads, n_agents, dim)
#     #         actions = self.sample_actions(available_actions)
#     #         (
#     #             new_obs,
#     #             new_share_obs,
#     #             rewards,
#     #             dones,
#     #             infos,
#     #             new_available_actions,
#     #         ) = self.envs.step(actions)
#     #         next_obs = new_obs.copy()
#     #         next_share_obs = new_share_obs.copy()
#     #         next_available_actions = new_available_actions.copy()
#     #         #print(obs.shape,"shape")#(20, 3, 18) shape
#     #         # print(obs.shape, "shape")#(20, 3) shape
#     #         # print(obs)
#     #         # print(share_obs.shape,"s2")#(20, 3, 3, 54) s2zhengc(20, 3, 54) s2
#     #         # print(share_obs)#
#     #         data = (
#     #             share_obs,
#     #             obs.transpose(1, 0, 2),
#     #             actions.transpose(1, 0, 2),
#     #             available_actions.transpose(1, 0, 2)
#     #             if len(np.array(available_actions).shape) == 3
#     #             else None,
#     #             rewards,
#     #             dones,
#     #             infos,
#     #             next_share_obs,
#     #             next_obs,
#     #             next_available_actions.transpose(1, 0, 2)
#     #             if len(np.array(available_actions).shape) == 3
#     #             else None,
#     #         )
#     #         self.insert(data)
#     #         obs = new_obs
#     #         share_obs = new_share_obs
#     #         available_actions = new_available_actions
#     #     return obs, share_obs, available_actions
#
#     def insert(self, data):
#         (
#             share_obs,  # (n_threads, n_agents, share_obs_dim)
#             obs,  # (n_agents, n_threads, obs_dim)
#             actions,  # (n_agents, n_threads, action_dim)
#             available_actions,  # None or (n_agents, n_threads, action_number)
#             rewards,  # (n_threads, n_agents, 1)
#             dones,  # (n_threads, n_agents)
#             infos,  ## type: list, shape: (n_threads, n_agents)
#             next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
#             next_obs,  # (n_threads, n_agents, next_obs_dim)
#             next_available_actions,  # None or (n_agents, n_threads, next_action_number)
#         ) = data
#
#         dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
#         reward_env = np.mean(rewards, axis=1).flatten()
#         self.train_episode_rewards += reward_env
#
#         # valid_transition denotes whether each transition is valid or not (invalid if corresponding agent is dead)
#         # shape: (n_threads, n_agents, 1)
#         valid_transitions = 1 - self.agent_deaths
#
#         self.agent_deaths = np.expand_dims(dones, axis=-1)
#
#         # terms use False to denote truncation and True to denote termination
#         if self.state_type == "EP":
#             terms = np.full((self.algo_args["train"]["n_rollout_threads"], 1), False)
#             for i in range(self.algo_args["train"]["n_rollout_threads"]):
#                 if dones_env[i]:
#                     if not (
#                         "bad_transition" in infos[i][0].keys()
#                         and infos[i][0]["bad_transition"] == True
#                     ):
#                         terms[i][0] = True
#         elif self.state_type == "FP":
#             terms = np.full(
#                 (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
#                 False,
#             )
#             for i in range(self.algo_args["train"]["n_rollout_threads"]):
#                 for agent_id in range(self.num_agents):
#                     if dones[i][agent_id]:
#                         if not (
#                             "bad_transition" in infos[i][agent_id].keys()
#                             and infos[i][agent_id]["bad_transition"] == True
#                         ):
#                             terms[i][agent_id][0] = True
#
#         for i in range(self.algo_args["train"]["n_rollout_threads"]):
#             if dones_env[i]:
#                 self.done_episodes_rewards.append(self.train_episode_rewards[i])
#                 self.train_episode_rewards[i] = 0
#                 self.agent_deaths = np.zeros(
#                     (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
#                 )
#                 if "original_obs" in infos[i][0]:
#                     next_obs[i] = infos[i][0]["original_obs"].copy()
#                 if "original_state" in infos[i][0]:
#                     next_share_obs[i] = infos[i][0]["original_state"].copy()
#
#         if self.state_type == "EP":
#             data = (
#                 share_obs[:, 0],  # (n_threads, share_obs_dim)
#                 obs,  # (n_agents, n_threads, obs_dim)
#                 actions,  # (n_agents, n_threads, action_dim)
#                 available_actions,  # None or (n_agents, n_threads, action_number)
#                 rewards[:, 0],  # (n_threads, 1)
#                 np.expand_dims(dones_env, axis=-1),  # (n_threads, 1)
#                 valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
#                 terms,  # (n_threads, 1)
#                 next_share_obs[:, 0],  # (n_threads, next_share_obs_dim)
#                 next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
#                 next_available_actions,  # None or (n_agents, n_threads, next_action_number)
#             )
#         elif self.state_type == "FP":
#             data = (
#                 share_obs,  # (n_threads, n_agents, share_obs_dim)
#                 obs,  # (n_agents, n_threads, obs_dim)
#                 actions,  # (n_agents, n_threads, action_dim)
#                 available_actions,  # None or (n_agents, n_threads, action_number)
#                 rewards,  # (n_threads, n_agents, 1)
#                 np.expand_dims(dones, axis=-1),  # (n_threads, n_agents, 1)
#                 valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
#                 terms,  # (n_threads, n_agents, 1)
#                 next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
#                 next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
#                 next_available_actions,  # None or (n_agents, n_threads, next_action_number)
#             )
#
#         self.buffer.insert(data)
#     ##预热阶段随机采样动作，
#     ##available_actions智能体可用的动作
#     def sample_actions(self, available_actions=None):
#         """Sample random actions for warmup.
#         Args:
#             available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
#                                  shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
#         Returns:
#             actions: (np.ndarray) sampled actions, shape is (n_threads, n_agents, dim)
#         """
#         actions = []
#         for agent_id in range(self.num_agents):
#             action = []
#             for thread in range(self.algo_args["train"]["n_rollout_threads"]):
#                 #print(self.algo_args["train"]["n_rollout_threads"])
#                 if available_actions[thread] is None:
#                     action.append(self.action_spaces[agent_id].sample())
#                 else:
#                     action.append(
#                         Categorical(
#                             torch.tensor(available_actions[thread, agent_id, :])
#                         ).sample()
#                     )
#             actions.append(action)
#             #print(action,"mnnnn")
#         # if self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
#         return np.expand_dims(np.array(actions).transpose(1, 0), axis=-1)
#
#         #return np.array(actions).transpose(1, 0, 2)
#
#     # def sample_actions(self, available_actions=None):
#     #     """Sample random actions for warmup.
#     #     Args:
#     #         available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
#     #                              shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
#     #     Returns:
#     #         actions: (np.ndarray) sampled actions, shape is (n_threads, n_agents, dim)
#     #     """
#     #     actions = []
#     #     #print(available_actions,"available_actions")
#     #     for agent_id in range(self.num_agents):
#     #         action = []
#     #         for thread in range(self.algo_args["train"]["n_rollout_threads"]):
#     #             if available_actions[thread] is None:
#     #                 action.append(self.action_spaces[agent_id].sample())
#     #             else:
#     #                 action.append(
#     #                     np.random.choice(self.action_spaces[agent_id])
#     #                 )
#     #                 # action.append(
#     #                 #     np.random.choice(self.action_spaces[agent_id], p=available_actions[thread, agent_id, :])
#     #                 # )
#     #                 # action.append(
#     #                 #     Categorical(
#     #                 #         torch.tensor(available_actions[thread, agent_id, :])
#     #                 #     ).sample()
#     #                 # )
#     #         actions.append(action)
#     #     # if self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
#     #     #     return np.expand_dims(np.array(actions).transpose(1, 0), axis=-1)
#     #     return np.expand_dims(np.array(actions).transpose(1, 0), axis=-1)
#     #     #return np.array(actions).transpose(1, 0, 2)
#     ##obs每个线程每个智能体观测值，为(n_threads, n_agents, dim)，add_random一个布尔值表示是否添加随机性
#     @torch.no_grad()
#     def get_actions(self, obs, available_actions=None, add_random=True):
#         """Get actions for rollout.
#         Args:
#             obs: (np.ndarray) input observation, shape is (n_threads, n_agents, dim)
#             available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
#                                  shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
#             add_random: (bool) whether to add randomness
#         Returns:
#             actions: (np.ndarray) agent actions, shape is (n_threads, n_agents, dim)
#         """
#         if self.args["algo"] == "hasac":
#             actions = []
#             for agent_id in range(self.num_agents):
#                 if (
#                     len(np.array(available_actions).shape) == 3
#                 ):  # (n_threads, n_agents, action_number)
#                     actions.append(
#                         _t2n(
#                             self.actor[agent_id].get_actions(
#                                 obs[:, agent_id],
#                                 available_actions[:, agent_id],
#                                 add_random,
#                             )
#                         )
#                     )
#                 else:  # (n_threads, ) of None
#                     actions.append(
#                         _t2n(
#                             self.actor[agent_id].get_actions(
#                                 obs[:, agent_id], stochastic=add_random
#                             )
#                         )
#                     )
#         else:
#             actions = []
#             for agent_id in range(self.num_agents):
#                 # print(available_actions[:,agent_id],"acccccccccc")
#                 # print(obs[:, agent_id],"testeeeeee")
#                 actions.append(
#                     _t2n(self.actor[agent_id].get_actions(obs[:, agent_id],available_actions[:,agent_id], add_random))
#                 )
#                 #print(actions,"acccccc")
#
#         return np.array(actions).transpose(1, 0, 2)
#     # def get_actions(self, obs, available_actions=None, add_random=True):
#     #     """Get actions for rollout.
#     #     Args:
#     #         obs: (np.ndarray) input observation, shape is (n_threads, n_agents, dim)
#     #         available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
#     #                              shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
#     #         add_random: (bool) whether to add randomness
#     #     Returns:
#     #         actions: (np.ndarray) agent actions, shape is (n_threads, n_agents, dim)
#     #     """
#     #     if self.args["algo"] == "hasac":
#     #         actions = []
#     #         for agent_id in range(self.num_agents):
#     #             if (
#     #                 len(np.array(available_actions).shape) == 3
#     #             ):  # (n_threads, n_agents, action_number)
#     #                 actions.append(
#     #                     _t2n(
#     #                         self.actor[agent_id].get_actions(
#     #                             obs[:, agent_id],
#     #                             available_actions[:, agent_id],
#     #                             add_random,
#     #                         )
#     #                     )
#     #                 )
#     #             else:  # (n_threads, ) of None
#     #                 actions.append(
#     #                     _t2n(
#     #                         self.actor[agent_id].get_actions(
#     #                             obs[:, agent_id], stochastic=add_random
#     #                         )
#     #                     )
#     #                 )
#     #     else:
#     #
#     #         actions = []
#     #         for agent_id in range(self.num_agents):
#     #             actions.append(
#     #                 self.actor[agent_id].get_actions(obs[:, agent_id], add_random)
#     #             )
#     #             #print(actions,"11")
#     #     return np.array(actions).transpose(1, 0, 2)
#
#     # def get_actions(self, obs, available_actions=None, add_random=True):
#     #     """Get actions for rollout.
#     #     Args:
#     #         obs: (np.ndarray) input observation, shape is (n_threads, n_agents, dim)
#     #         available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
#     #                              shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
#     #         add_random: (bool) whether to add randomness
#     #     Returns:
#     #         actions: (np.ndarray) agent actions, shape is (n_threads, n_agents, dim)
#     #     """
#     #     if self.args["algo"] == "hasac":
#     #         actions = []
#     #         for agent_id in range(self.num_agents):
#     #             if (
#     #                 len(np.array(available_actions).shape) == 3
#     #             ):  # (n_threads, n_agents, action_number)
#     #                 actions.append(
#     #                     _t2n(
#     #                         self.actor[agent_id].get_actions(
#     #                             obs[:, agent_id],
#     #                             available_actions[:, agent_id],
#     #                             add_random,
#     #                         )
#     #                     )
#     #                 )
#     #             else:  # (n_threads, ) of None
#     #                 actions.append(
#     #                     _t2n(
#     #                         self.actor[agent_id].get_actions(
#     #                             obs[:, agent_id], stochastic=add_random
#     #                         )
#     #                     )
#     #                 )
#     #     else:
#     #         actions = []
#     #         for agent_id in range(self.num_agents):
#     #             ##
#     #             obs_agent = torch.from_numpy(np.stack(obs)[:, agent_id, :]).float().to(self.device)
#     #             actions.append(
#     #                 self.actor[agent_id].get_actions(obs_agent, self.envs.action_space[agent_id])
#     #             )
#     #             # actions.append(
#     #             #     _t2n(self.actor[agent_id].get_actions(obs_agent, self.envs.action_space[agent_id]))
#     #             # )
#     #             # actions.append(
#     #             #     _t2n(self.actor[agent_id].get_actions(obs[:, agent_id], add_random))
#     #             # )
#     #     return np.expand_dims(np.array(actions).transpose(1, 0), axis=-1)
#     #     #return np.array(actions).transpose(1, 0, 2)##返回数组表示每个线程中每个智能体的动作
#
#     def train(self):
#         """Train the model"""
#         raise NotImplementedError
#
#     @torch.no_grad()
#     def eval(self, step):
#         """Evaluate the model"""
#         eval_episode_rewards = []
#         one_episode_rewards = []
#         for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
#             one_episode_rewards.append([])
#             eval_episode_rewards.append([])
#         eval_episode = 0
#         if "smac" in self.args["env"]:
#             eval_battles_won = 0
#         if "football" in self.args["env"]:
#             eval_score_cnt = 0
#         episode_lens = []
#         one_episode_len = np.zeros(
#             self.algo_args["eval"]["n_eval_rollout_threads"], dtype=np.int_
#         )
#
#         eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
#
#         while True:
#             eval_actions = self.get_actions(
#                 eval_obs, available_actions=eval_available_actions, add_random=False
#             )
#             (
#                 eval_obs,
#                 eval_share_obs,
#                 eval_rewards,
#                 eval_dones,
#                 eval_infos,
#                 eval_available_actions,
#             ) = self.eval_envs.step(eval_actions)
#             for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
#                 one_episode_rewards[eval_i].append(eval_rewards[eval_i])
#
#             one_episode_len += 1
#
#             eval_dones_env = np.all(eval_dones, axis=1)
#
#             for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
#                 if eval_dones_env[eval_i]:
#                     eval_episode += 1
#                     if "smac" in self.args["env"]:
#                         if "v2" in self.args["env"]:
#                             if eval_infos[eval_i][0]["battle_won"]:
#                                 eval_battles_won += 1
#                         else:
#                             if eval_infos[eval_i][0]["won"]:
#                                 eval_battles_won += 1
#                     if "football" in self.args["env"]:
#                         if eval_infos[eval_i][0]["score_reward"] > 0:
#                             eval_score_cnt += 1
#                     eval_episode_rewards[eval_i].append(
#                         np.sum(one_episode_rewards[eval_i], axis=0)
#                     )
#                     one_episode_rewards[eval_i] = []
#                     episode_lens.append(one_episode_len[eval_i].copy())
#                     one_episode_len[eval_i] = 0
#
#             if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
#                 # eval_log returns whether the current model should be saved
#                 eval_episode_rewards = np.concatenate(
#                     [rewards for rewards in eval_episode_rewards if rewards]
#                 )
#                 eval_avg_rew = np.mean(eval_episode_rewards)
#                 eval_avg_len = np.mean(episode_lens)
#                 if "smac" in self.args["env"]:
#                     print(
#                         "Eval win rate is {}, eval average episode rewards is {}, eval average episode length is {}.".format(
#                             eval_battles_won / eval_episode, eval_avg_rew, eval_avg_len
#                         )
#                     )
#                 elif "football" in self.args["env"]:
#                     print(
#                         "Eval score rate is {}, eval average episode rewards is {}, eval average episode length is {}.".format(
#                             eval_score_cnt / eval_episode, eval_avg_rew, eval_avg_len
#                         )
#                     )
#                 else:
#                     print(
#                         f"Eval average episode reward is {eval_avg_rew}, eval average episode length is {eval_avg_len}.\n"
#                     )
#                 if "smac" in self.args["env"]:
#                     self.log_file.write(
#                         ",".join(
#                             map(
#                                 str,
#                                 [
#                                     step,
#                                     eval_avg_rew,
#                                     eval_avg_len,
#                                     eval_battles_won / eval_episode,
#                                 ],
#                             )
#                         )
#                         + "\n"
#                     )
#                 elif "football" in self.args["env"]:
#                     self.log_file.write(
#                         ",".join(
#                             map(
#                                 str,
#                                 [
#                                     step,
#                                     eval_avg_rew,
#                                     eval_avg_len,
#                                     eval_score_cnt / eval_episode,
#                                 ],
#                             )
#                         )
#                         + "\n"
#                     )
#                 else:
#                     self.log_file.write(
#                         ",".join(map(str, [step, eval_avg_rew, eval_avg_len])) + "\n"
#                     )
#                 self.log_file.flush()
#                 self.writter.add_scalar(
#                     "eval_average_episode_rewards", eval_avg_rew, step
#                 )
#                 self.writter.add_scalar(
#                     "eval_average_episode_length", eval_avg_len, step
#                 )
#                 break
#
#     @torch.no_grad()
#     def render(self):
#         """Render the model"""
#         print("start rendering")
#         if self.manual_expand_dims:
#             # this env needs manual expansion of the num_of_parallel_envs dimension
#             for _ in range(self.algo_args["render"]["render_episodes"]):
#                 eval_obs, _, eval_available_actions = self.envs.reset()
#                 eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
#                 eval_available_actions = np.array([eval_available_actions])
#                 rewards = 0
#                 while True:
#                     eval_actions = self.get_actions(
#                         eval_obs,
#                         available_actions=eval_available_actions,
#                         add_random=False,
#                     )
#                     (
#                         eval_obs,
#                         _,
#                         eval_rewards,
#                         eval_dones,
#                         _,
#                         eval_available_actions,
#                     ) = self.envs.step(eval_actions[0])
#                     rewards += eval_rewards[0][0]
#                     eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
#                     eval_available_actions = np.array([eval_available_actions])
#                     if self.manual_render:
#                         self.envs.render()
#                     if self.manual_delay:
#                         time.sleep(0.1)
#                     if eval_dones[0]:
#                         print(f"total reward of this episode: {rewards}")
#                         break
#         else:
#             # this env does not need manual expansion of the num_of_parallel_envs dimension
#             # such as dexhands, which instantiates a parallel env of 64 pair of hands
#             for _ in range(self.algo_args["render"]["render_episodes"]):
#                 eval_obs, _, eval_available_actions = self.envs.reset()
#                 rewards = 0
#                 while True:
#                     eval_actions = self.get_actions(
#                         eval_obs,
#                         available_actions=eval_available_actions,
#                         add_random=False,
#                     )
#                     (
#                         eval_obs,
#                         _,
#                         eval_rewards,
#                         eval_dones,
#                         _,
#                         eval_available_actions,
#                     ) = self.envs.step(eval_actions)
#                     rewards += eval_rewards[0][0][0]
#                     if self.manual_render:
#                         self.envs.render()
#                     if self.manual_delay:
#                         time.sleep(0.1)
#                     if eval_dones[0][0]:
#                         print(f"total reward of this episode: {rewards}")
#                         break
#         if "smac" in self.args["env"]:  # replay for smac, no rendering
#             if "v2" in self.args["env"]:
#                 self.envs.env.save_replay()
#             else:
#                 self.envs.save_replay()
#
#     def restore(self):
#         """Restore the model"""
#         for agent_id in range(self.num_agents):
#             self.actor[agent_id].restore(self.algo_args["train"]["model_dir"], agent_id)
#         if not self.algo_args["render"]["use_render"]:
#             self.critic.restore(self.algo_args["train"]["model_dir"])
#             if self.value_normalizer is not None:
#                 value_normalizer_state_dict = torch.load(
#                     str(self.algo_args["train"]["model_dir"])
#                     + "/value_normalizer"
#                     + ".pt"
#                 )
#                 self.value_normalizer.load_state_dict(value_normalizer_state_dict)
#
#     def save(self):
#         """Save the model"""
#         for agent_id in range(self.num_agents):
#             self.actor[agent_id].save(self.save_dir, agent_id)
#         self.critic.save(self.save_dir)
#         if self.value_normalizer is not None:
#             torch.save(
#                 self.value_normalizer.state_dict(),
#                 str(self.save_dir) + "/value_normalizer" + ".pt",
#             )
#
#     def close(self):
#         """Close environment, writter, and log file."""
#         # post process
#         if self.algo_args["render"]["use_render"]:
#             self.envs.close()
#         else:
#             self.envs.close()
#             if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
#                 self.eval_envs.close()
#             self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
#             self.writter.close()
#             self.log_file.close()
