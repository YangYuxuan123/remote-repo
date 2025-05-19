import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

#处理离散动作空间中的动作选择&采样
##接收一个批量的logits和一个epsilon值，使用epsilon贪婪策略返回一个one-hot样本;如果epsilon为0，将返回最大logit对应的one-hot向量，否则将以epsilon概率返回一个随机的one-hot向量
def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(
        torch.eye(logits.shape[1])[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ],
        requires_grad=False,
    )
    # chooses between best and random actions using epsilon greedy
    return torch.stack(
        [
            argmax_acs[i] if r > eps else rand_acs[i]
            for i, r in enumerate(torch.rand(logits.shape[0]))
        ]
    )

##这个函数从 Gumbel(0, 1) 分布中采样。Gumbel 分布常用于采样自离散分布。
def sample_gumbel(shape, device, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

##这个函数从 Gumbel-Softmax 分布中采样。Gumbel-Softmax 分布是 softmax 分布的一个扩展，它允许从多项分布中进行可微分的采样
def gumbel_softmax_sample(logits, temperature, device):
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data), device=device)
    return F.softmax(y / temperature, dim=1)

##这个函数从 Gumbel-Softmax 分布中采样，并可以选择是否离散化。如果 hard 参数为 True，那么它将返回 one-hot 向量，否则它将返回一个概率分布。
def gumbel_softmax(logits, device, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, device=device)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y
