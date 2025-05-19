"""MADDPG algorithm."""
from harl.algorithms.actors.haddpg import HADDPG


class MADDPG(HADDPG):
    pass
##maddpg里面epsilon参数并没有被包含在内可能是因为在maddpg算法实现中并没有使用到epsilon参数
##强化学习中epsilon通常用于e-greedy策略用于控制探索和利用的平衡，在maddpg算法中通常使用的是添加噪声的方式进行探索而不是e-greedy策略

##epsilon是一个用于控制探索-利用均衡的参数，具体来说是一个概率值用于决定在选择动作时是进行探索还是利用
##在 get_actions 和 get_target_actions 方法中，如果一个随机生成的数小于 epsilon，那么就会随机选择一个动作（探索）；否则，就会选择当前策略认为最优的动作（利用）。这种方法被称为 ε-greedy 策略。
##MADDPG摸鱼这个参数可能因为maddpg使用了不同的探索策略


