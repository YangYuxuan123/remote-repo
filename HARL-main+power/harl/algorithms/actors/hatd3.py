"""HATD3 algorithm."""
import numpy as np
import torch
from torch.distributions import Categorical

from harl.utils.envs_tools import check
from harl.algorithms.actors.haddpg import HADDPG
from harl.utils.discrete_util import gumbel_softmax_sample

class HATD3(HADDPG):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, act_space, device)
        self.policy_noise = args["policy_noise"]
        self.noise_clip = args["noise_clip"]
        self.epsilon=args["epsilon"]

    def get_target_actions(self, obs):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        state = check(obs).to(**self.tpdv)
        actions = self.actor(state)
        # device = logits.device
        #
        # actions = gumbel_softmax_sample(logits, temperature=1.0, device=device)
        #actions = gumbel_softmax_sample(logits, temperature=1.0, device=torch.device("cuda"))
        # print(actions,"gum")
        if np.random.random() < self.epsilon:
            action = torch.randint(
                low=0, high=18, size=(*state.shape[:-1], 1)
            )
        else:
            action = actions.argmax(dim=-1, keepdim=True)
        # action = action.tolist()
        # action = torch.tensor(action, dtype=torch.float32)  # * self.expl_noise
        #
        # action.requires_grad_()

        return action
