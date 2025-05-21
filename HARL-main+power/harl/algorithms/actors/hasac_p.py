"""HASAC algorithm."""
import torch
from harl.models.policy_models.squashed_gaussian_policy import SquashedGaussianPolicy
from harl.models.policy_models.multibranch_stochastic_policy import MultiBranchStochasticPolicy
from harl.models.policy_models.stochastic_mlp_policy import StochasticMlpPolicy
from harl.utils.discrete_util import gumbel_softmax
from harl.utils.envs_tools import check
from harl.algorithms.actors.off_policy_base import OffPolicyBase
from torch.distributions import Categorical

class HASAC(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.device = device
        self.action_type = act_space.__class__.__name__
        self.act_space = act_space

        if act_space.__class__.__name__ == "Dict":  # 参数化动作空间（多分支）
            assert "channel" in act_space.spaces and "power" in act_space.spaces
            self.actor = MultiBranchStochasticPolicy(args, obs_space, act_space, device)
        elif act_space.__class__.__name__ == "Box":
            self.actor = SquashedGaussianPolicy(args, obs_space, act_space, device)
        else:
            self.actor = StochasticMlpPolicy(args, obs_space, act_space, device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)  # 为策略网络（actor）创建一个 Adam 优化器
        self.turn_off_grad()

    def get_actions(self, obs, available_actions=None, stochastic=True):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)  #从ndarry转为tensor了
        if self.action_type == "Dict":
            actions = self.actor(obs, available_actions, stochastic)  # 直接委托给actor处理
        else:
            # 原有逻辑（Box/Discrete）
            if self.action_type == "Box":
                actions, _ = self.actor(obs, stochastic=stochastic, with_logprob=False)
            else:
                actions = self.actor(obs, available_actions, stochastic)
        return actions

    def get_actions_with_logprobs(self, obs, available_actions=None, stochastic=True):
        """Get actions and logprobs of actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (batch_size, dim)
            logp_actions: (torch.Tensor) log probabilities of actions taken by this actor, shape is (batch_size, 1)
        """
        obs = check(obs).to(**self.tpdv)
        if self.action_type == "Dict":
            logits_dict = self.actor.get_logits(obs, available_actions)
            # 选择动作（采样 or argmax）
            if stochastic:
                channel_action = Categorical(logits=logits_dict["channel"]).sample()
                power_action = Categorical(logits=logits_dict["power"]).sample()
            else:
                channel_action = logits_dict["channel"].argmax(dim=-1)
                power_action = logits_dict["power"].argmax(dim=-1)
            # 获取选中动作对应的得分（logit）
            channel_logit = logits_dict["channel"].gather(-1, channel_action.unsqueeze(-1)).squeeze(-1)
            power_logit = logits_dict["power"].gather(-1, power_action.unsqueeze(-1)).squeeze(-1)
            # 掩码无效的功率动作
            mask = (channel_action < self.act_space["channel"].n - 1).float()
            power_logit = power_logit * mask

            return {"channel": channel_action, "power": power_action}, channel_logit + power_logit

        else:
            if self.action_type == "Box":
                actions, logp_actions = self.actor(
                    obs, stochastic=stochastic, with_logprob=True
                )
            elif self.action_type == "Discrete":
                logits = self.actor.get_logits(obs, available_actions)  # get_logits返回一个未归一化的概率分布（即 logits），表示每个动作的得分。
                actions = gumbel_softmax(
                    logits, hard=True, device=self.device
                )  # onehot actions
                logp_actions = torch.sum(actions * logits, dim=-1, keepdim=True)
            elif self.action_type == "MultiDiscrete":
                logits = self.actor.get_logits(obs, available_actions)
                actions = []
                logp_actions = []
                for logit in logits:
                    action = gumbel_softmax(
                        logit, hard=True, device=self.device
                    )  # onehot actions
                    logp_action = torch.sum(action * logit, dim=-1, keepdim=True)
                    actions.append(action)
                    logp_actions.append(logp_action)
                actions = torch.cat(actions, dim=-1)
                logp_actions = torch.cat(logp_actions, dim=-1)
            return actions, logp_actions  # 所选动作的数字为分数，其余动作为0

    def save(self, save_dir, id):
        """Save the actor."""
        torch.save(
            self.actor.state_dict(), str(save_dir) + "/actor_agent" + str(id) + ".pt"
        )

    def restore(self, model_dir, id):
        """Restore the actor."""
        actor_state_dict = torch.load(str(model_dir) + "/actor_agent" + str(id) + ".pt")
        self.actor.load_state_dict(actor_state_dict)
