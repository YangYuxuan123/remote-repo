"""Tools for HARL."""
import numpy as np
import torch


# def _t2n(value):
#     """Convert torch.Tensor to numpy.ndarray."""
#     return value.detach().cpu().numpy()
'''def _t2n(value):
    """Convert torch.Tensor, Dict, or other types to numpy.ndarray or Python lists."""
    if isinstance(value, torch.distributions.Categorical):
        value = value.probs
    elif isinstance(value, list):
        return np.array(value)
    elif isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value  # 其他类型（如int/float）直接返回'''

def _t2n(value):
    """Convert torch.Tensor or torch.distributions.Categorical to numpy.ndarray."""
    if isinstance(value, torch.distributions.Categorical):
        value = value.probs
    elif isinstance(value,list):
        return np.array(value)
    return value.detach().cpu().numpy()

def _flatten(T, N, value):
    """Flatten the first two dimensions of a tensor."""
    return value.reshape(T * N, *value.shape[2:])
    #将张量的前两个维度展平（合并），而保持其余维度不变。

def _sa_cast(value):
    """This function is used for buffer data operation.
    Specifically, it transposes a tensor from (episode_length, n_rollout_threads, *dim) to (n_rollout_threads, episode_length, *dim).
    Then it combines the first two dimensions into one dimension.
    """
    return value.transpose(1, 0, 2).reshape(-1, *value.shape[2:])


def _ma_cast(value):
    """This function is used for buffer data operation.
    Specifically, it transposes a tensor from (episode_length, n_rollout_threads, num_agents, *dim) to (n_rollout_threads, num_agents, episode_length, *dim).
    Then it combines the first three dimensions into one dimension.
    """
    return value.transpose(1, 2, 0, 3).reshape(-1, *value.shape[3:])
#转置顺序