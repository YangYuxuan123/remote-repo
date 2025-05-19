"""Tools for HARL."""
import numpy as np
import torch


# def _t2n(value):
#     """Convert torch.Tensor to numpy.ndarray."""
#     return value.detach().cpu().numpy()
'''def _t2n(value):
    """Convert torch.Tensor or torch.distributions.Categorical to numpy.ndarray."""
    if isinstance(value, torch.distributions.Categorical):
        value = value.probs
    elif isinstance(value,list):
        return np.array(value)
    elif isinstance(value, dict):  # 处理 Dict 类型
        return {key: _t2n(val) for key, val in value.items()}  # 递归处理字典中的每个值
    return value.detach().cpu().numpy()'''

def _t2n(value):
    """Convert torch.Tensor, Dict, or other types to numpy.ndarray or Python lists."""
    if isinstance(value, torch.distributions.Categorical):
        value = value.probs
    elif isinstance(value, list):
        return np.array(value)
    elif isinstance(value, dict):
        # 特殊处理：当字典包含'channel'和'power'时，转换为列表 [channel_val, power_val]
        '''if 'channel' in value and 'power' in value:
            channel_val = value['channel'].item() if isinstance(value['channel'], torch.Tensor) else value['channel']
            power_val = value['power'].item() if isinstance(value['power'], torch.Tensor) else value['power']
            return [channel_val, power_val]'''

        if 'channel' in value and 'power' in value:
            ch = value['channel']
            pw = value['power']
            if isinstance(ch, torch.Tensor):
                ch = ch.detach().cpu().numpy()
            if isinstance(pw, torch.Tensor):
                pw = pw.detach().cpu().numpy()
            # 合并为 [[ch[0], pw[0]], [ch[1], pw[1]], ...]
            return np.stack((ch, pw), axis=1).tolist()

        # 普通字典递归处理
        return {key: _t2n(val) for key, val in value.items()}
    elif isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value  # 其他类型（如int/float）直接返回

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