"""Runner for off-policy HARL algorithms."""
import torch
import numpy as np
import torch.nn.functional as F
from harl.utils.discrete_util import gumbel_softmax_sample
from harl.runners.off_policy_base_runner import OffPolicyBaseRunner

class OffPolicyHARunner(OffPolicyBaseRunner):
    """Runner for off-policy HA algorithms."""

    def train(self):
        """Train the model"""
        self.total_it += 1  #记录update_num更新的计数器
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        # train critic
        self.critic.turn_on_grad()
        if self.args["algo"] == "hasac":
            next_actions = []
            next_logp_actions = []
            for agent_id in range(self.num_agents):
                next_action, next_logp_action = self.actor[
                    agent_id
                ].get_actions_with_logprobs(
                    sp_next_obs[agent_id],
                    sp_next_available_actions[agent_id]
                    if sp_next_available_actions is not None
                    else None,
                )
                next_actions.append(next_action)
                next_logp_actions.append(next_logp_action)
            ## 训练Critic的部分
                # sp_share_obs,共享观察，在某些多智能体环境中，所有智能体可能会共享一部分或全部的观察信息；这个变量存储的就是这部分共享的观察信息
                # sp_actions,每个智能体动作，其中每个元素对应一个智能体动作
                # sp_reward,每个智能体奖励，每个元素对应一个智能体奖励
                # sp_done,每个智能体完成状态是一个列表，其中每个元素对应一个智能体完成状态
                # sp_valid_transition,#每个智能体有效转移状态，是一个列表其中每个元素对应一个智能体有效转移状态，如果智能体状态转移有效的
                # sp_term,每个智能体终止状态，每个元素对应一个智能体终止状态，如果智能体任务终止
                # sp_next_share_obs,#下一个时间步共享观察，与sp_share_obs类似对应下一时间步观察信息
                # next_actions,下一个时间步每个智能体动作，这是一个列表每个元素对应一个智能体在下一个观察状态下的动作
                # next_logp_actions,下一个时间步每个智能体动作的对数概率，是一个列表每个元素对应一个智能体在下一个观察状态下的动作对数概率
                # sp_gamma,每个智能体折扣因子，是一个列表其中每个元素对应一个智能体折扣因子
                # self.value_normalizer,值归一化器，用于归一化预测值的工具，帮助稳定训练过程
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_valid_transition,
                sp_term,
                sp_next_share_obs,
                next_actions,
                next_logp_actions,
                sp_gamma,
                self.value_normalizer,
            )
        else:
            next_actions = []
            for agent_id in range(self.num_agents):
                next_actions.append(
                    self.actor[agent_id].get_target_actions(sp_next_obs[agent_id])
                )
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
            )
        self.critic.turn_off_grad()
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)
        if self.total_it % self.policy_freq == 0:
            # train actors
            if self.args["algo"] == "hasac":
                actions = []
                logp_actions = []
                with torch.no_grad():
                    for agent_id in range(self.num_agents):
                        action, logp_action = self.actor[
                            agent_id
                        ].get_actions_with_logprobs(
                            sp_obs[agent_id],
                            sp_available_actions[agent_id]
                            if sp_available_actions is not None
                            else None,
                        )
                        actions.append(action)
                        logp_actions.append(logp_action)
                # actions shape: (n_agents, batch_size, dim)
                # logp_actions shape: (n_agents, batch_size, 1)
                if self.fixed_order:
                    agent_order = list(range(self.num_agents))
                else:
                    agent_order = list(np.random.permutation(self.num_agents))
                for agent_id in agent_order:
                    self.actor[agent_id].turn_on_grad()
                    # train this agent
                    actions[agent_id], logp_actions[agent_id] = self.actor[
                        agent_id
                    ].get_actions_with_logprobs( 
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                    if self.state_type == "EP":
                        logp_action = logp_actions[agent_id]
                        processed = []
                        for action_dict in actions:
                            channel = action_dict["channel"].squeeze(-1)  # (batch_size,)
                            power = action_dict["power"].squeeze(-1)  # (batch_size,)

                            channel_onehot = F.one_hot(channel, num_classes=26)  # -> (batch_size, 26)
                            power_onehot = F.one_hot(power, num_classes=4)  # -> (batch_size, 4)

                            mask = (channel != 25).unsqueeze(-1).float()  # -> (batch_size, 1)
                            power_onehot = power_onehot * mask  # -> (batch_size, 4)
                            # 3) 拼成 (batch_size, 30)
                            full = torch.cat([channel_onehot, power_onehot], dim=-1)  # -> (batch_size, 30)
                            processed.append(full)

                        actions_stack = torch.stack(processed, dim=0)  # -> (n_agents, batch_size, 30)
                        n_agents, batch_size, act_dim = actions_stack.shape
                        actions_t = (
                            actions_stack
                            .permute(1, 0, 2)  # -> (batch_size, n_agents, act_dim)
                            .reshape(batch_size, n_agents * act_dim)  # -> (batch_size, n_agents*act_dim)
                        )
                        #actions_t = torch.cat(actions, dim=-1)
                    elif self.state_type == "FP":
                        logp_action = torch.tile(
                            logp_actions[agent_id], (self.num_agents, 1)
                        )
                        actions_t = torch.tile(
                            torch.cat(actions, dim=-1), (self.num_agents, 1)
                        )
                    value_pred = self.critic.get_values(sp_share_obs, actions_t)
                    ##haddpg no;hatd3 no
                    if self.algo_args["algo"]["use_policy_active_masks"]:
                        if self.state_type == "EP":
                            actor_loss = (
                                -torch.sum(
                                    (value_pred - self.alpha[agent_id] * logp_action)
                                    * sp_valid_transition[agent_id]
                                )
                                / sp_valid_transition[agent_id].sum()
                            )
                        elif self.state_type == "FP":
                            valid_transition = torch.tile(
                                sp_valid_transition[agent_id], (self.num_agents, 1)
                            )
                            actor_loss = (
                                -torch.sum(
                                    (value_pred - self.alpha[agent_id] * logp_action)
                                    * valid_transition
                                )
                                / valid_transition.sum()
                            )
                    else:
                        actor_loss = -torch.mean(
                            value_pred - self.alpha[agent_id] * logp_action
                        )
                    self.actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor[agent_id].actor_optimizer.step()
                    self.actor[agent_id].turn_off_grad()
                    # train this agent's alpha
                    if self.algo_args["algo"]["auto_alpha"]:
                        log_prob = (
                            logp_actions[agent_id].detach()
                            + self.target_entropy[agent_id]
                        )
                        alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                        self.alpha_optimizer[agent_id].zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer[agent_id].step()
                        self.alpha[agent_id] = torch.exp(
                            self.log_alpha[agent_id].detach()
                        )
                    actions[agent_id], _ = self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                # train critic's alpha
                if self.algo_args["algo"]["auto_alpha"]:
                    self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))
            else:
                if self.args["algo"] == "had3qn":
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, 1)
                    update_actions, get_values = self.critic.train_values(
                        sp_share_obs, actions
                    )
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # actor preds
                        actor_values = self.actor[agent_id].train_values(
                            sp_obs[agent_id], actions[agent_id]
                        )
                        # critic preds
                        critic_values = get_values()
                        # update
                        actor_loss = torch.mean(F.mse_loss(actor_values, critic_values))
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        update_actions(agent_id)
                else:
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, dim)
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # train this agent
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )

                        actions_t = torch.cat(actions, dim=-1)

                        value_pred = self.critic.get_values(sp_share_obs, actions_t)
                        value_pred = gumbel_softmax_sample(value_pred, temperature=1.0, device=self.device)
                        #print(value_pred,"d")
                        # value_pred=np.argmax(value_pred, axis=1)
                        # print(sp_share_obs)
                        # print(actions_t,"test7777")
                        actor_loss = -torch.mean(value_pred)
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                # soft update
                for agent_id in range(self.num_agents):
                    self.actor[agent_id].soft_update()
            self.critic.soft_update()
