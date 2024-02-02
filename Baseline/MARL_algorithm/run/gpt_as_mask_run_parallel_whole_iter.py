import datetime
import glob
import os
import re
import threading
import time
import copy
from os.path import abspath, dirname
from types import SimpleNamespace as SN
import pandas as pd
import numpy as np
import torch
import pdb
import random
import wandb
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from components.reward_scaler import RewardScaler
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
import sys
from utils.logging import Logger
from utils.timehelper import time_left_gpt, time_str
from utils.extract_task_code import *
import openai
import time
import re
import importlib
import multiprocessing
from multiprocessing import Pipe, Process, Pool
from collections import defaultdict
import yaml
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from Baseline.MARL_algorithm.utils.logging import Logger, get_logger

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
openai.api_type = "azure"
openai.api_base = "openai-api-base"
openai.api_version = "openai-api-version"
openai.api_key = "openai-api-key"


# import matplotlib.pyplot as plt
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    # args.device = "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    tmp_config = {k: _config[k] for k in _config if k != "env_args"}
    tmp_config.update(
        {f"env_agrs.{k}": _config["env_args"][k] for k in _config["env_args"]}
    )
    print(
        pd.Series(tmp_config, name="HyperParameter Value")
        .transpose()
        .sort_index()
        .fillna("")
        .to_markdown()
    )

    # configure tensorboard logger
    ts = datetime.now().strftime("%m%dT%H%M")
    unique_token = f"{_config['name']}_{_config['env_args']['n_agents']}_{_config['env_args']['task_type']}_seed{_config['seed']}_{ts}"
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger, _config=_config)
    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def gpt_generate_mask(messages, checker_messages, checker_feedback, gpt_generator_feedback, logger, args,
                      gpt_update_iter, execution_error_feedback, mask_output_dir):
    err_list = []
    successes = [False for i in range(args.gpt_sample_num)]
    sys.path.append(mask_output_dir)
    os.makedirs(os.path.join(mask_output_dir, "code"), exist_ok=True)
    sys.path.append(os.path.join(mask_output_dir, "code"))
    if os.getenv("AMLT_OUTPUT_DIR") is not None:
        time_postfix = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        amlt_code_read_dir = os.path.join(os.getenv("CODE_DIR_IN_CONTAINER"), time_postfix)
        os.makedirs(amlt_code_read_dir, exist_ok=True)
        sys.path.append(amlt_code_read_dir)
    while not np.any(successes):

        gpt_generate_messages = copy.deepcopy(messages)
        if len(err_list) > 0:
            gpt_generate_messages[-1]['content'] += execution_error_feedback.format(traceback_msg=",".join(err_list))
        err_list = []
        mask_func_list = []
        responses = []

        with Pool(args.gpt_sample_num) as p:
            worker_input = [(thread_id, gpt_generate_messages, checker_messages, checker_feedback,
                             gpt_generator_feedback, mask_output_dir, gpt_update_iter)
                            for thread_id in range(args.gpt_sample_num)]
            result = p.starmap(worker, worker_input)
        successes, responses = zip(*result)
        successes = list(successes)
        if not np.all(successes):
            logger.console_logger.info("Code terminated due to too many failed attempts!")
            return None
        for response_id in range(args.gpt_sample_num):
            response_cur = responses[response_id]["message"]["content"]
            logger.console_logger.info(f"Iteration {gpt_update_iter}: Processing Code Run {response_id}")
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
            try:
                gpt_mask_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                logger.console_logger.info(
                    f"Iteration {gpt_update_iter}: Code Run {response_id} cannot parse function signature due to \n {str(e)}")
                successes[response_id] = False
                mask_func_list.append(None)
                err_list.append(str(e))
                continue

            output_file = os.path.join(mask_output_dir, "code",
                                       f"iter{gpt_update_iter}_response{response_id}_maskonly.py")

            with open(output_file, 'w') as file:
                file.writelines("from ReplenishmentEnv.env.agent_states import AgentStates" + '\n')
                file.writelines("from ReplenishmentEnv.env.supply_chain import SupplyChain" + '\n')
                file.writelines("import numpy as np" + '\n')
                file.writelines("import torch" + '\n')
                file.writelines("from torch import Tensor" + '\n')
                file.writelines(code_string + '\n')

            if os.getenv("AMLT_OUTPUT_DIR") is not None:
                with open(os.path.join(amlt_code_read_dir, f"iter{gpt_update_iter}_response{response_id}_maskonly.py"),
                          'w') as file:
                    file.writelines("from ReplenishmentEnv.env.agent_states import AgentStates" + '\n')
                    file.writelines("from ReplenishmentEnv.env.supply_chain import SupplyChain" + '\n')
                    file.writelines("import numpy as np" + '\n')
                    file.writelines("import torch" + '\n')
                    file.writelines("from torch import Tensor" + '\n')
                    file.writelines(code_string + '\n')
            # module_name = "Baseline.MARL_algorithm.mask." + f"iter{gpt_update_iter}_response{response_id}_maskonly"
            module_name = f"iter{gpt_update_iter}_response{response_id}_maskonly"
            func_name = gpt_mask_signature.split('(')[0]
            try:
                time.sleep(0.5)
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    module = importlib.reload(module)
                else:
                    module = importlib.import_module(module_name)
                mask_func = getattr(module, func_name)
                mask_func_list.append(mask_func)
                successes[response_id] = True
            except Exception as e:
                err_list.append(str(e))
                logger.console_logger.info(f"Exploration function {response_id} can not be loaded due to \n{str(e)}")
                mask_func_list.append(None)
                continue
    return responses, mask_func_list, successes


def get_pipe(gpt_sample_num):
    stop_flag = multiprocessing.Value('b', False)
    parent_conns, worker_conns = zip(
        *[Pipe() for _ in range(gpt_sample_num)]
    )
    ps = [
        Process(
            target=runner_worker,
            args=((worker_conn, stop_flag),),
        )
        for worker_conn in worker_conns]
    for p in ps:
        p.daemon = False
        p.start()

    return parent_conns, worker_conns, ps


def divide_arr_into_interval(arr, interval):
    divide_list = [[] for i in range(len(interval))]
    for i, itv in enumerate(interval):
        picked_v = np.where((arr >= itv[0]) & (arr <= itv[1]))
        for v in zip(*picked_v):
            divide_list[i].append(v)
    dic = {}
    for i in range(len(divide_list)):
        for j in range(len(divide_list[i])):
            index_0 = divide_list[i][j][0]
            if index_0 not in dic:
                dic[index_0] = [[] for _ in range(len(divide_list))]
            dic[index_0][i].append(divide_list[i][j])
    divide_list = list(dic.values())

    return divide_list


def assign_gpu(subthread_id):
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available")
    num_gpus = torch.cuda.device_count()
    gpu_id = subthread_id % num_gpus
    target_device = torch.device(f'cuda:{gpu_id}')
    return gpu_id, target_device


def run_sequential(args, logger, _config=None):
    # multiprocessing.set_start_method('fork', force=True)
    multiprocessing.set_start_method('spawn', force=True)
    # multiprocessing.set_start_method('forkserver', force=True)
    if "azureml" in sys.modules:
        logging.getLogger("azureml").setLevel(logging.WARNING)
    wandb.run.define_metric("episode")
    wandb.define_metric("train_step")
    wandb.define_metric("train reward", step_metric="train_step")
    wandb.define_metric("evaluate_step")
    wandb.define_metric("val reward", step_metric="evaluate_step")
    wandb.define_metric("test reward", step_metric="evaluate_step")
    wandb.define_metric("plot_step")
    wandb.define_metric("action available frequency", step_metric="plot_step")
    wandb.define_metric("action chosen percentage", step_metric="plot_step")
    set_seed(args.seed)
    env_info = r_REGISTRY[args.runner](args=args, logger=logger).get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "mean_action": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.int,
        },
        "probs": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "reward": {"vshape": (1,)},
        "individual_rewards": {"vshape": (1,), "group": "agents"},
        "cur_balance": {"vshape": (1,), "group": "agents"},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    prompt_dir = f'Baseline/MARL_algorithm/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    initial_system_checker = file_to_string(f'{prompt_dir}/initial_system_checker.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    initial_checker = file_to_string(f'{prompt_dir}/initial_checker.txt')
    exploration_signature = file_to_string(f'{prompt_dir}/exploration_signature.txt')
    policy_feedback_stat = file_to_string(f'{prompt_dir}/policy_feedback_stat.txt')
    policy_feedback_list = file_to_string(f'{prompt_dir}/policy_feedback_list.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
    state_definition = file_to_string(f'{prompt_dir}/state_definition.txt')
    supply_chain_definition = file_to_string(f'{prompt_dir}/supply_chain_definition.txt')
    action_space_str = file_to_string(f'{prompt_dir}/action_space.txt')
    transition_definition = file_to_string(f'{prompt_dir}/transition_definition.txt')
    task_introduction = file_to_string(f'{prompt_dir}/task_introduction.txt')
    generate_error_feedback = file_to_string(f'{prompt_dir}/generating_error_feedback.txt')
    mask_feedback = file_to_string(f'{prompt_dir}/mask_feedback.txt')
    reward_definition = file_to_string(f'{prompt_dir}/reward_definition.txt')
    mask_feedback_discrete = file_to_string(f'{prompt_dir}/mask_feedback_discrete.txt')
    checker_feedback = file_to_string(f'{prompt_dir}/checker_feedback.txt')
    action_chosen_percentage_feedback = file_to_string(f'{prompt_dir}/action_chosen_percentage_feedback.txt')
    gpt_generator_feedback = file_to_string(f'{prompt_dir}/gpt_generator_feedback.txt')

    initial_system = initial_system.format(task_exploration_signature_string=exploration_signature) + code_output_tip
    initial_system_checker = initial_system_checker.format(task_exploration_signature_string=exploration_signature)
    initial_user = initial_user.format(task_introduction=task_introduction, transition_definition=transition_definition,
                                       state_definition=state_definition,
                                       supply_chain_definition=supply_chain_definition,
                                       reward_definition=reward_definition, action_space=action_space_str)
    initial_checker = initial_checker.format(task_introduction=task_introduction,
                                             transition_definition=transition_definition,
                                             state_definition=state_definition,
                                             supply_chain_definition=supply_chain_definition,
                                             reward_definition=reward_definition, action_space=action_space_str,
                                             code_output_tip=code_output_tip)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]
    checker_messages = [{"role": "system", "content": initial_system_checker},
                        {"role": "user", "content": initial_checker}]
    total_messages = messages.copy()
    print(pd.DataFrame(scheme).transpose().sort_index().fillna("").to_markdown())

    # Start training
    last_update_ep = 0
    gpt_update_iter = 0
    start_time = time.time()
    last_time = start_time
    checkpoint_update_ep = 0
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    now = datetime.now()
    time_postfix = now.strftime("%Y%m%d_%H_%M_%S")
    mask_output_dir = os.path.join("Baseline", "MARL_algorithm", "mask", args.run,
                                   args.env_args['task_type'] + time_postfix) if os.getenv(
        "AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "mask")
    logger.console_logger.info(f"Exploration function and interaction stored in {mask_output_dir}")

    os.makedirs(mask_output_dir, exist_ok=True)
    plot_outputs_path = os.path.join(mask_output_dir, "plot_outputs")
    os.makedirs(plot_outputs_path, exist_ok=True)

    # gpt_err_list = []
    legend = []
    last_save_path = None
    begin_time = time.time()
    while gpt_update_iter < args.iters:
        iter_begin_time = time.time()
        val_reward_component_stats_list = []
        test_reward_component_stats_list = []
        total_mask_distribution_list = []
        mask_component_distribution_list = []
        action_chosen_times_list = []
        performance_list = []
        profit_list = []
        training_curve_log_list = []
        test_return_4_wandb_list = []
        val_return_4_wandb_list = []
        log_time_list = []
        # generate or load mask function
        if args.load_mask_func == False or gpt_update_iter > 0:
            logger.console_logger.info("Begin to generate exploration function.")
            result = gpt_generate_mask(messages, checker_messages, checker_feedback, gpt_generator_feedback, logger,
                                       args,
                                       gpt_update_iter, generate_error_feedback, mask_output_dir)
            if len(result) == 1:
                logger.console_logger.info("Finished Training")
                return
            responses, mask_func_list, successes = result
            logger.console_logger.info("Exploration function generation ends.")
        else:
            logger.console_logger.info("Load mask function from disk.")
            dir_path, module_name = args.checkpoint_maskfunc_path.rsplit('/', 1)
            module_name = module_name.split('.')[0]
            sys.path.append(dir_path)
            try:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    module = importlib.reload(module)
                else:
                    module = importlib.import_module(module_name)
                mask_func = getattr(module, "compute_mask")
                mask_func_list = [mask_func for i in range(args.gpt_sample_num)]
                successes = [True for i in range(args.gpt_sample_num)]
                response = file_to_string(args.checkpoint_maskfunc_path)
                response = "def" + response.split('def', 1)[1]
                responses = [{"message": {"content": response}} for i in range(args.gpt_sample_num)]
            except Exception as e:
                logger.console_logger.info(f"Mask function can not be loaded due to \n{str(e)}")
                logger.console_logger.info("Exiting...")
                return

        recv_list = []
        # get pipe
        # stop_flag = multiprocessing.Value('b', False)
        stop_event = multiprocessing.Event()
        manager = multiprocessing.Manager()
        shared_expected_duration_dict = manager.dict({i: 0 for i in range(args.gpt_sample_num)})
        shared_completed_task = manager.list([False for i in range(args.gpt_sample_num)])
        shared_successes_task = manager.list([False for i in range(args.gpt_sample_num)])
        parent_conns, worker_conns = zip(
            *[Pipe() for _ in range(args.gpt_sample_num)]
        )
        ps = [
            Process(
                target=runner_worker,
                args=(
                worker_conn, stop_event, shared_expected_duration_dict, shared_completed_task, shared_successes_task),
            )
            for worker_conn in worker_conns]
        for p in ps:
            p.daemon = False
            p.start()
        # parent_conns, worker_conns, ps = get_pipe(args.gpt_sample_num)

        for idx, parent_conn in enumerate(parent_conns):
            parent_conn.send(["setup", (
            args, scheme, groups, preprocess, env_info, mask_func_list[idx], gpt_update_iter, mask_output_dir, idx,
            last_save_path)])
        for parent_conn in parent_conns:
            parent_conn.recv()

        for parent_conn in parent_conns:
            parent_conn.send(["run_iter", None])

        while False in shared_completed_task:
            min_expected_duration = np.inf
            for subthread_id in range(args.gpt_sample_num):
                if shared_completed_task[subthread_id] and shared_successes_task[subthread_id]:
                    min_expected_duration = min(min_expected_duration, shared_expected_duration_dict[subthread_id])

            max_allowed_duration = min_expected_duration * 1.5
            for subthread_id in range(args.gpt_sample_num):
                if shared_completed_task[subthread_id] == False:
                    if shared_expected_duration_dict[subthread_id] > max_allowed_duration:
                        successes[subthread_id] = False
                        logger.console_logger.info(
                            f"subthread {subthread_id} terminated because it's expected complete time {shared_expected_duration_dict[subthread_id]} exceeds the max allowed duration {max_allowed_duration}!")
                        shared_expected_duration_dict[subthread_id] = np.inf
                        shared_completed_task[subthread_id] = True
            time.sleep(1)

        print("All sub-processes have completed.")

        # Step 1: Run for a whole episode at a time
        for idx, parent_conn in enumerate(parent_conns):
            recv = parent_conn.recv()
            recv_list.append(recv)

        successes = [recv[0] for recv in recv_list]
        if not np.any(successes):
            logger.console_logger.info("All threads fail. Retry...")
            for idx, parent_conn in enumerate(parent_conns):
                try:
                    parent_conn.send(("close", None))
                    parent_conn.close()
                    ps[idx].terminate()
                    ps[idx].join()
                except OSError as e:
                    logger.console_logger.info(f"Subthread {idx} is already terminated")
                    continue
            continue
        logger.console_logger.info(
            f"Total time elapsed : {time_str(time.time() - begin_time)}, time elapsed for this iter : {time_str(time.time() - iter_begin_time)}, alive threads : {sum(successes)}")
        for idx, success in enumerate(successes):
            if success:
                test_return_4_wandb_list.append(recv_list[idx][1][0])
                val_return_4_wandb_list.append(recv_list[idx][1][1])
                performance_list.append(recv_list[idx][1][2])
                training_curve_log_list.append(recv_list[idx][1][3])
                val_reward_component_stats_list.append(recv_list[idx][1][4])
                test_reward_component_stats_list.append(recv_list[idx][1][5])
                profit_list.append(recv_list[idx][1][6])
                total_mask_distribution_list.append(recv_list[idx][1][7])
                mask_component_distribution_list.append(recv_list[idx][1][8])
                action_chosen_times_list.append(recv_list[idx][1][9])
                log_time_list.append(recv_list[idx][1][10])
            else:
                test_return_4_wandb_list.append(None)
                val_return_4_wandb_list.append(None)
                performance_list.append(-np.inf)
                training_curve_log_list.append(None)
                val_reward_component_stats_list.append(None)
                test_reward_component_stats_list.append(None)
                profit_list.append(None)
                total_mask_distribution_list.append(None)
                mask_component_distribution_list.append(None)
                action_chosen_times_list.append(None)
                log_time_list.append(None)

        best_sample_idx = np.argmax(performance_list)
        eval_len = min(5, np.array(val_reward_component_stats_list[best_sample_idx]['reward']).shape[0])
        val_best_performance = np.array(val_reward_component_stats_list[best_sample_idx]['reward'])[
                               -eval_len:].sum() / eval_len
        test_best_performance = np.array(test_reward_component_stats_list[best_sample_idx]['reward'])[
                                -eval_len:].sum() / eval_len
        best_content = ""

        log_time = log_time_list[best_sample_idx]
        training_curve_log = training_curve_log_list[best_sample_idx]
        for i, cur in enumerate(training_curve_log):
            cur.update({"episode": log_time["train_curve"][i]})
            wandb.log(cur)
        wandb_dict = {}
        for k in val_reward_component_stats_list[best_sample_idx].keys():
            wandb_dict.update({
                f"mean test {k}": np.array(test_reward_component_stats_list[best_sample_idx][k])[
                                  -eval_len:].mean(axis=0).sum(),
                f"mean val {k}": np.array(val_reward_component_stats_list[best_sample_idx][k])[-eval_len:].mean(
                    axis=0).sum()})
        wandb_dict.update({"episode": (1 + gpt_update_iter) * args.iter_nepisode})
        wandb.log(wandb_dict)

        train_return_4_wandb_list = profit_list[best_sample_idx]
        test_return_4_wandb_list = np.sum(test_reward_component_stats_list[best_sample_idx]['reward'],
                                          axis=1).tolist()
        val_return_4_wandb_list = np.sum(val_reward_component_stats_list[best_sample_idx]['reward'],
                                         axis=1).tolist()
        action_available_frequency_hist = np.array(total_mask_distribution_list[best_sample_idx]).mean(
            axis=(0, 1, 2)).tolist()
        action_chosen_percentage_hist = (
                np.array(action_chosen_times_list[best_sample_idx]).sum(axis=0) / np.array(
            action_chosen_times_list[best_sample_idx]).sum(axis=-1, keepdims=True).sum(axis=0)).mean(
            axis=0).tolist()
        for i, cur in enumerate(train_return_4_wandb_list):
            wandb.log({"train reward": train_return_4_wandb_list[i], "train_step": log_time["train"][i]})
        for i, cur in enumerate(test_return_4_wandb_list):
            wandb.log({"val reward": val_return_4_wandb_list[i], "test reward": test_return_4_wandb_list[i],
                       "evaluate_step": log_time["evaluate"][i]})
        parent_conns[best_sample_idx].send(("get_action_space", None))
        action_space = parent_conns[best_sample_idx].recv()
        action_space_str = [str(action) for action in action_space]
        plt.bar(np.arange(args.n_actions), action_available_frequency_hist)
        plt.xticks(ticks=range(len(action_space)), labels=action_space_str, rotation='vertical')
        plt.title("Action available frequency")
        plt.ylim([0, 1])
        wandb.log({"action available frequency": wandb.Image(plt), "plot_step": gpt_update_iter})
        plt.close()

        plt.bar(np.arange(args.n_actions), action_chosen_percentage_hist)
        plt.xticks(ticks=range(len(action_space)), labels=action_space_str, rotation='vertical')
        plt.title("Action chosen percentage")
        plt.ylim([0, 1])
        wandb.log({"action chosen percentage": wandb.Image(plt), "plot_step": gpt_update_iter})
        plt.close()

        with open(os.path.join(plot_outputs_path, f"action_available_frequency_hist_iter{gpt_update_iter}.pkl"),
                  'wb') as f:
            pickle.dump(action_available_frequency_hist, f)
        with open(os.path.join(plot_outputs_path, f"action_chosen_percentage_hist_iter{gpt_update_iter}.pkl"),
                  'wb') as f:
            pickle.dump(action_chosen_percentage_hist, f)

        logger.console_logger.info(
            f"Current iter {gpt_update_iter}, best sample idx {best_sample_idx}, best val performance {val_best_performance}, best test performance {test_best_performance}")
        best_total_mask_distribution = np.array(total_mask_distribution_list[best_sample_idx]).mean(axis=(0, 2))
        # best_mask_component_list = [mask_component]
        if None not in mask_component_distribution_list[best_sample_idx]:
            best_mask_component_distribution = {
                k: np.mean([d.get(k, 0) for d in mask_component_distribution_list[best_sample_idx]],
                           axis=(0, 2))
                for k in set.union(*[set(d) for d in mask_component_distribution_list[best_sample_idx]])
            }
        else:
            best_mask_component_distribution = None
        action_chosen_percentage = np.array(action_chosen_times_list[best_sample_idx]).sum(axis=0) / np.array(
            action_chosen_times_list[best_sample_idx]).sum(axis=-1, keepdims=True).sum(axis=0)
        best_content += policy_feedback_stat.format(epoch_freq=str(args.evaluate_nepisode))
        for metric_cur, v in val_reward_component_stats_list[best_sample_idx].items():
            v = np.array(v)
            early_stage = np.mean(v[:eval_len], axis=0)
            late_stage = np.mean(v[-eval_len:], axis=0)
            max_ep_idx = np.argmax(v.sum(axis=1))
            min_ep_idx = np.argmin(v.sum(axis=1))
            metric_cur_max = v[max_ep_idx]
            metric_cur_min = v[min_ep_idx]
            metric_cur_mean = v.mean(axis=0)
            best_content += f"metric_name: {metric_cur}, Max: {np.array2string(metric_cur_max)}, Min: {np.array2string(metric_cur_min)}, Mean in the early training stage: {np.array2string(early_stage)}, Mean in the late training stage: {np.array2string(late_stage)}, Mean in all the training stage: {np.array2string(metric_cur_mean)}\n"
        action_available_frequency_interval = [[0, 0.3], [0.3, 0.7], [0.7, 1]]
        expected_percentage = round(1 / len(action_space), 2)
        action_chosen_percentage_interval = [[0, 0.5 * expected_percentage],
                                             [0.5 * expected_percentage, 1.5 * expected_percentage],
                                             [1.5 * expected_percentage, 3 * expected_percentage],
                                             [3 * expected_percentage, 1]]
        total_action_available_frequency_divide_list = divide_arr_into_interval(best_total_mask_distribution,
                                                                                action_available_frequency_interval)
        action_chosen_percentage_divide_list = divide_arr_into_interval(action_chosen_percentage,
                                                                        action_chosen_percentage_interval)
        best_content += mask_feedback_discrete + "\n"
        best_content += "For total mask,\n"
        for i, sub_divide_list in enumerate(total_action_available_frequency_divide_list):
            best_content += f"For warehouse {i},"
            for j, action_list in enumerate(sub_divide_list):
                best_content += f"The action for action available frequency in interval {str(action_available_frequency_interval[j])} is: "
                if len(action_list) > 0:
                    for action in action_list:
                        best_content += f"{str(action_space[action[1]])} "
                else:
                    best_content += "No action in this interval"
                best_content += "\n"
            best_content += "\n"
        if best_mask_component_distribution is not None:
            for mask_name, mask_value in best_mask_component_distribution.items():
                best_content += f"For mask component {str(mask_name)},\n"
                component_action_available_frequency_divide_list = divide_arr_into_interval(mask_value,
                                                                                            action_available_frequency_interval)
                for i, sub_divide_list in enumerate(component_action_available_frequency_divide_list):
                    best_content += f"For warehouse {i},"
                    for j, action_list in enumerate(sub_divide_list):
                        best_content += f"The action for action available frequency in interval {str(action_available_frequency_interval[j])} is: "
                        if len(action_list) > 0:
                            for action in action_list:

                                try:
                                    best_content += f"{str(action_space[action[1]])} "
                                except Exception as e:
                                    print("best_mask_component_distribution: ",
                                          best_mask_component_distribution)
                                    print("component_action_available_frequency_divide_list: ",
                                          component_action_available_frequency_divide_list)
                                    print("sub_divide_list: ", sub_divide_list)
                                    print("action_list: ", action_list)
                                    print("action: ", action)
                                    raise ValueError(str(e))
                        else:
                            best_content += "No action in this interval"
                        best_content += "\n"
                    best_content += "\n"
                best_content += "\n"

        best_content += action_chosen_percentage_feedback
        for i, sub_divide_list in enumerate(action_chosen_percentage_divide_list):
            best_content += f"For warehouse {i},"
            for j, action_list in enumerate(sub_divide_list):
                best_content += f"The action for action chosen percentage in interval {str(action_chosen_percentage_interval[j])} is: "
                if len(action_list) > 0:
                    for action in action_list:
                        best_content += f"{str(action_space[action[1]])} "
                else:
                    best_content += "No action in this interval"
                best_content += "\n"
            best_content += "\n"

        checker_messages = [{"role": "system", "content": initial_system_checker},
                            {"role": "user",
                             "content": initial_checker + "The old exploration function and its analisis is:\n" +
                                        responses[best_sample_idx]["message"]["content"] + best_content}]

        best_content += code_feedback
        if len(messages) == 2:
            messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}
            messages[-1] = {"role": "user", "content": best_content}

        total_messages.append(
            {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]})
        total_messages.append({"role": "user", "content": best_content})
        output_file = os.path.join(mask_output_dir, "total_messages.txt")
        with open(output_file, "w") as file:
            for message in total_messages:
                file.write(message["role"] + "\n")
                file.write(message[
                               "content"] + "\n------------------------------------------------------------------------------------\n\n")
        parent_conns[best_sample_idx].send(("run_visualize", None))
        parent_conns[best_sample_idx].recv()
        logger.console_logger.info(f"Finish visualizing")
        parent_conns[best_sample_idx].send(("save", None))
        last_save_path = parent_conns[best_sample_idx].recv()
        logger.console_logger.info(f"Finish saving")
        for idx, parent_conn in enumerate(parent_conns):
            try:
                parent_conn.send(("close", None))
                parent_conn.close()
                ps[idx].terminate()
                ps[idx].join()
            except OSError as e:
                logger.console_logger.info(f"Subthread {idx} is already terminated")
                continue

        gpt_update_iter += 1


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not torch.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
                                          config["test_nepisode"] // config["batch_size_run"]
                                  ) * config["batch_size_run"]

    return config


def worker(thread_id, gpt_generate_messages, checker_messages, checker_feedback, gpt_generator_feedback,
           mask_output_dir, gpt_update_iter):

    total_messages = copy.deepcopy(checker_messages)
    gpt_generate_messages_init_len = len(gpt_generate_messages)
    mask_output_dir = os.path.join(mask_output_dir, "checker_messages", str(gpt_update_iter))
    os.makedirs(mask_output_dir, exist_ok=True)
    output_file = os.path.join(mask_output_dir, f"thread{thread_id}_total_messages.txt")
    try:
        for i in range(5):
            for attempt in range(1000):
                try:
                    response = openai.ChatCompletion.create(
                        engine="gpt-4-32k",
                        messages=gpt_generate_messages,
                        temperature=0.7,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None
                    )
                    break
                except Exception as e:
                    if attempt % 10 == 0:
                        print("thread {} gpt generator generate attempt {} fails, trying again".format(thread_id,
                                                                                                       attempt))
                    time.sleep(1)
            if response is None:
                return (False, None)
            response_content = response["choices"][-1]["message"]["content"]
            gpt_checker_messages = copy.deepcopy(checker_messages)
            # gpt_checker_messages[-1]['content'] = gpt_checker_messages[-1]['content'].replace('{gpt_response}', response_content)
            gpt_checker_messages[-1]['content'] = gpt_checker_messages[-1]['content'] + gpt_generator_feedback.format(
                gpt_response=response_content)

            for attempt in range(1000):
                try:
                    response_check = openai.ChatCompletion.create(
                        engine="gpt-4-32k",
                        messages=gpt_checker_messages,
                        temperature=0.7,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None
                    )
                    break
                except Exception as e:
                    if attempt % 10 == 0:
                        print(
                            "thread {} gpt checker generate attempt {} fails, trying again".format(thread_id, attempt))
                    time.sleep(1)
            if response_check is None:
                return (False, None)
            response_checker_content = response_check["choices"][-1]["message"]["content"]
            gpt_checker_feedback = checker_feedback.format(checker_feedback=response_checker_content)
            total_messages.append({"role": "gpt", "content": response_content})
            total_messages.append({"role": "checker", "content": gpt_checker_feedback})
            with open(output_file, "w") as file:
                for message in total_messages:
                    file.write(message["role"] + "\n")
                    file.write(message[
                                   "content"] + "\n------------------------------------------------------------------------------------\n\n")

            if "Code passes check" in response_checker_content:
                break
            else:
                print(
                    "---------------------thread {} trial {} fails to pass checker, trying again---------------------".format(
                        thread_id, i))
                if len(gpt_generate_messages) == gpt_generate_messages_init_len:
                    gpt_generate_messages += [{"role": "assistant", "content": response_content}]
                    gpt_generate_messages += [{"role": "user", "content": gpt_checker_feedback}]
                else:
                    assert len(gpt_generate_messages) == gpt_generate_messages_init_len + 2
                    gpt_generate_messages[-2] = {"role": "assistant", "content": response_content}
                    gpt_generate_messages[-1] = {"role": "user", "content": gpt_checker_feedback}

        print(f"\n---------------------thread {thread_id} success generates code---------------------\n")
        return (True, response['choices'][-1])
    except Exception as e:
        print(f"thread {thread_id} fails to generate code due to {e}")
        return (False, None)


def runner_worker(remote, stop_event, shared_expected_duration_dict, shared_completed_task, shared_successes_task):
    torch.set_num_threads(1)
    try:
        while True:
            cmd, data = remote.recv()
            if stop_event.is_set():
                raise RuntimeError("CUDA out of memory")
            if cmd == "run_iter":

                try:
                    result = runner.run_iter(shared_expected_duration_dict, shared_completed_task,
                                             shared_successes_task)
                    shared_completed_task[subthread_id] = True
                    remote.send(result)  # TODO:send这个之后显存占用量显著增加？为什么会这样？
                except Exception as e:
                    shared_completed_task[subthread_id] = True
                    str_e = str(e).lower()
                    if "out of memory" in str_e:
                        stop_event.set()
                    logger.console_logger.info(
                        f"Iteration {current_iter}, thread {subthread_id} fails because of {str(e)}")
                    result = [False, str(e)]
                    remote.send(result)
                    runner.close_env()
                    test_runner.close_env()
                    val_runner.close_env()
                    remote.close()
                    break

            elif cmd == "setup":
                logger = Logger(get_logger())
                args, scheme, groups, preprocess, env_info, mask_func, current_iter, mask_output_dir, subthread_id, load_path = data
                if args.device == 'cuda':
                    gpu_id, target_device = assign_gpu(subthread_id)
                    args.device = target_device
                    args.gpu_id = gpu_id
                logger.console_logger.info(f"Thread : {subthread_id}, use gpu id : {gpu_id}")
                runner = r_REGISTRY[args.runner](args=args, logger=logger)
                test_args = copy.deepcopy(args)
                test_args.env_args["mode"] = "test"
                val_args = copy.deepcopy(args)
                val_args.env_args["mode"] = "validation"
                buffer = ReplayBuffer(
                    scheme,
                    groups,
                    args.buffer_size,
                    env_info["episode_limit"] + 1,
                    preprocess=preprocess,
                    device="cpu" if args.buffer_cpu_only else args.device)
                mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
                learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
                if current_iter > 1e-5:
                    learner.load_models(load_path)
                    with open(os.path.join(load_path, "buffer.pkl"), 'rb') as f:
                        buffer = pickle.load(f)
                    with open(os.path.join(load_path, "runner_para.pkl"), 'rb') as f:
                        runner.t_env, runner.last_update_ep, runner.last_evaluate_ep = pickle.load(f)

                if args.device != 'cpu':
                    learner.cuda(args.gpu_id)
                val_runner = r_REGISTRY["gpt_test_episode"](args=val_args, logger=logger)
                test_runner = r_REGISTRY["gpt_test_episode"](args=test_args, logger=logger)
                test_runner.setup(scheme, groups, preprocess, learner.mac)
                val_runner.setup(scheme, groups, preprocess, learner.mac)
                runner.setup(scheme, groups, preprocess, learner.mac, mask_func, learner, buffer, test_runner,
                             val_runner, current_iter, subthread_id)
                remote.send(None)
            elif cmd == "get_action_space":
                remote.send(runner.action_space)
            elif cmd == "save":
                save_path = os.path.join(mask_output_dir, f"iter{current_iter}_best")
                os.makedirs(save_path, exist_ok=True)
                runner.learner.save_models(save_path)
                with open(os.path.join(save_path, "buffer.pkl"), 'wb') as f:
                    pickle.dump(runner.buffer, f)
                with open(os.path.join(save_path, "runner_para.pkl"), 'wb') as f:
                    pickle.dump((runner.t_env, runner.last_update_ep, runner.last_evaluate_ep), f)

                if os.getenv("AMLT_OUTPUT_DIR") is not None:
                    amlt_save_path = os.path.join(os.getenv("CODE_DIR_IN_CONTAINER"), f"iter{current_iter}_best")
                    sys.path.append(amlt_save_path)
                    os.makedirs(amlt_save_path, exist_ok=True)
                    runner.learner.save_models(amlt_save_path)
                    with open(os.path.join(amlt_save_path, "buffer.pkl"), 'wb') as f:
                        pickle.dump(runner.buffer, f)
                    with open(os.path.join(amlt_save_path, "runner_para.pkl"), 'wb') as f:
                        pickle.dump((runner.t_env, runner.last_update_ep, runner.last_evaluate_ep), f)
                remote.send(save_path if os.getenv("AMLT_OUTPUT_DIR") is None else amlt_save_path)
            elif cmd == "run_visualize":
                visual_outputs_path = os.path.join(mask_output_dir, "visual_outputs")
                logger.console_logger.info(f"Saving visualizations to {visual_outputs_path}/{current_iter}")
                os.makedirs(os.path.join(visual_outputs_path, str(current_iter)), exist_ok=True)
                test_runner.run_visualize(visual_outputs_path, current_iter)
                remote.send(None)
            elif cmd == "close":
                runner.close_env()
                test_runner.close_env()
                val_runner.close_env()
                remote.close()
                break
            else:
                raise NotImplementedError
    except Exception as e:
        logger.console_logger.info(
            f"Iteration {current_iter}, thread {subthread_id} fails because of {str(e)}. Thread {subthread_id} exit. ")
        if remote:
            remote.close()
        if runner:
            runner.close_env()
        if test_runner:
            test_runner.close_env()
        if val_runner:
            val_runner.close_env()
        sys.exit(0)