"""Tools for loading and updating configs."""
import time
import os
import json
import yaml
from uu import Error


def get_defaults_yaml_args(algo, env):
    """Load config file for user-specified algo and env.
    Args:
        algo: (str) Algorithm name.
        env: (str) Environment name.
    Returns:
        algo_args: (dict) Algorithm config.
        env_args: (dict) Environment config.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "configs", "envs_cfgs", f"{env}.yaml")

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)
    return algo_args, env_args

#更新配置参数，接收一个未解析的参数字典和一个或多个要更新的参数字典，将未解析的参数更新到这些参数字典中
#根据环境的类型和环境的参数生成任务的名称
def update_args(unparsed_dict, *args):
    """Update loaded config with unparsed command-line arguments.
    Args:
        unparsed_dict: (dict) Unparsed command-line arguments.
        *args: (list[dict]) argument dicts to be updated.
    """

    def update_dict(dict1, dict2):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(dict1, dict2[k])
            else:
                if k in dict1:
                    dict2[k] = dict1[k]

    for args_dict in args:
        update_dict(unparsed_dict, args_dict)


def get_task_name(env, env_args):
    """Get task name."""
    if env == "smac":
        task = env_args["map_name"]
    elif env == "smacv2":
        task = env_args["map_name"]
    elif env == "mamujoco":
        task = f"{env_args['scenario']}-{env_args['agent_conf']}"
    elif env == "pettingzoo_mpe":
        if env_args["continuous_actions"]:
            task = f"{env_args['scenario']}-continuous"
        else:
            task = f"{env_args['scenario']}-discrete"
    elif env == "gym":
        task = env_args["scenario"]
    elif env == "football":
        task = env_args["env_name"]
    elif env == "dexhands":
        task = env_args["task"]
    elif env == "lag":
        task = f"{env_args['scenario']}-{env_args['task']}"
    elif env == "DSA":
        task = env_args["env_name"]

        # if env_args["continuous_actions"]:
        #     task = f"{env_args['scenario']}-continuous"
        # else:
        #     task = f"{env_args['scenario']}-discrete"
    return task

#初始化用于保存结果的目录，返回结果路径、日志路径和一个 SummaryWriter 对象。
def init_dir(env, env_args, algo, exp_name, seed, logger_path):
    """Init directory for saving results."""
    task = get_task_name(env, env_args)
    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    results_path = os.path.join(
        logger_path,
        env,
        task,
        algo,
        exp_name,
        "-".join(["seed-{:0>5}".format(seed), hms_time]),
    )
    log_path = os.path.join(results_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    from tensorboardX import SummaryWriter

    writter = SummaryWriter(log_path)##将信息写入TensorBoard可读的日志文件
    models_path = os.path.join(results_path, "models")
    os.makedirs(models_path, exist_ok=True)
    return results_path, log_path, models_path, writter

#检查一个值是否可用被序列化为json
def is_json_serializable(value):
    """Check if v is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except Error:
        return False

#将一个对象转换为可以被序列化为JSON的格式
def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)

#保存程序的配置，将主参数、算法参数和环境参数保存到一个JSON文件中
def save_config(args, algo_args, env_args, run_dir):
    """Save the configuration of the program."""
    config = {"main_args": args, "algo_args": algo_args, "env_args": env_args}
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as out:
        out.write(output)
