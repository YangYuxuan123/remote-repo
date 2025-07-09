# -*- coding: utf-8 -*-
"""Train an algorithm."""
import warnings
warnings.filterwarnings('ignore')
import argparse#处理命令行参数&选项；接口调用
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1]))

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# device = torch.device("cpu")
#args.n_gpu = torch.cuda.device_count()
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from certifi.__main__ import args

#with torch.device("/gpu:0"):
     #sess.run(tf.global_variables_initializer())

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="maddpg", #改
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )

    parser.add_argument(
        "--env",
        type=str,
        default="DSA",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
            "DSA",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag,DSA.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )#友好命令行接口；命令行选项名称：提供参数值
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )

    #args命名空间
    #args：例如，如果你在命令行中运行 python script.py --exp_name my_experiment，那么 args.exp_name 的值就会是 "my_experiment"。
    #unparsed_args：
    #args.algo 的值将是 "happo" args.env 的值将是 "smac" args.exp_name 的值将是 "test"
    args, unparsed_args = parser.parse_known_args()

    print(args)#Namespace(algo='happo', env='pettingzoo_mpe', exp_name='installtest', load_config='')
    print(unparsed_args)#[]
    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    #提取参数名称：keys = [k[2:] for k in unparsed_args[0::2]]：这行代码从 unparsed_args 中提取出每个参数的名称。unparsed_args[0::2] 表示从 unparsed_args 中取出索引为偶数的元素，即所有的参数名称。k[2:] 是用来去除参数名称前的 --
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    #print(keys)#[]
    #这行代码从 unparsed_args 中提取出每个参数的值。unparsed_args[1::2] 表示从 unparsed_args 中取出索引为奇数的元素，即所有的参数值。process(v) 是一个函数，它尝试将参数值作为 Python 表达式进行求值，并返回结果
    values = [process(v) for v in unparsed_args[1::2]]
    #print(values)#[]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    #print(unparsed_dict)#{}
    args = vars(args)  # convert to dict#这行代码将 args 对象（一个由 argparse 返回的命名空间对象）转换为字典。vars() 函数返回对象的 __dict__ 属性，这个属性包含了对象的所有属性及其值。  在这个例子中，args 对象包含了所有已知的命令行参数及其值。例如，如果你在命令行中运行 python script.py --exp_name my_experiment，那么 args.exp_name 的值就会是 "my_experiment"。  通过将 args 对象转换为字典，你可以更方便地访问和操作命令行参数。例如，你可以使用字典的索引操作（如 args["exp_name"]）来访问参数的值，而不是使用属性访问操作（如 args.exp_name）。
    ##这段代码的目的是允许用户通过命令行选项或配置文件来设置训练参数，这提供了一种灵活的方式来配置训练过程。
    #load函数的返回值是反序列化后的 Python 对象。JSON文档反序列化为python对象，接受一个支持.read()方法的文件类对象fp，该对象包含一个json文档
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    ##更新1训练参数
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line
    #这段代码检查命令行参数 env 是否等于 "dexhands"。如果等于，那么它将导入 isaacgym 模块。#这个环境无关
    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch#机器人学习模拟环境库

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    #这段代码是在设置训练参数。它首先检查命令行参数 env 是否等于 "dexhands"。如果等于，那么它会修改 algo_args 字典中的一些值。
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()#
    runner.close()


if __name__ == "__main__":
    main()
