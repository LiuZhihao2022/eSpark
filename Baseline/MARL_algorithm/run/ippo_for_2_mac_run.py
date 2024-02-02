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

from utils.logging import Logger
from utils.timehelper import time_left, time_str
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
    ts = datetime.datetime.now().strftime("%m%dT%H%M")
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
    run_sequential(args=args, logger=logger)

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


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    runner.env_info['n_agents'] = int(runner.env_info['n_agents']/2)
    runner.env_info['state_shape'] = int(runner.env_info['state_shape']/2)
    # Set up schemes and groups here
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    set_seed(args.seed)
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

    buffer_1 = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
    buffer_2 = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
    logger.console_logger.info("MDP Components:")
    print(pd.DataFrame(buffer_1.scheme).transpose().sort_index().fillna("").to_markdown())

    # Setup multiagent controller here
    mac_1 = mac_REGISTRY[args.mac](buffer_1.scheme, groups, args)
    mac_2 = mac_REGISTRY[args.mac](buffer_2.scheme, groups, args)
            
    val_args = copy.deepcopy(args)
    val_args.env_args["mode"] = "validation"
    val_runner = r_REGISTRY[args.runner](args=val_args, logger=logger)

    test_args = copy.deepcopy(args)
    test_args.env_args["mode"] = "test"
    test_runner = r_REGISTRY[args.runner](args=test_args, logger=logger)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac_1 = mac_1, mac_2 = mac_2)
    val_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac_1 = mac_1, mac_2 = mac_2, set_stock_levels = runner.set_stock_levels)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac_1 = mac_1, mac_2 = mac_2, set_stock_levels = runner.set_stock_levels)

    if args.visualize:
        visual_runner = r_REGISTRY["episode_for_2_mac"](args=test_args, logger=logger)
        visual_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac_1=mac_1, mac_2=mac_2, set_stock_levels = runner.set_stock_levels)

    # Learner
    learner_1 = le_REGISTRY[args.learner](mac_1, buffer_1.scheme, logger, args)
    learner_2 = le_REGISTRY[args.learner](mac_2, buffer_2.scheme, logger, args)

    # Reward scaler
    reward_scaler = RewardScaler()

    if args.use_cuda:
        learner_1.cuda()
        learner_2.cuda()
    # TODO:这里也需要改成两个mac的形式
    if args.checkpoint_path:
        visual_runner.mac.load_models(args.checkpoint_path, postfix = '_best')
        vis_save_path = os.path.join(
            args.local_results_path, args.unique_token, "vis"
        ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "vis")
        logger.console_logger.info("Visualized result saved in {}".format(vis_save_path))
        visual_runner.run_visualize(visualize_path=vis_save_path, t="")
        logger.console_logger.info("Finish visualizing")
        return

    # Start training
    episode = 0
    last_test_T = 0
    last_log_T = 0
    model_save_time = 0
    visual_time = 0
    val_best_return = -np.inf

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # Pre-collect samples to fit reward scaler
    # 没有开这个参数，不用管
    if args.use_reward_normalization:
        episode_batch, train_stats, _, train_old_return = runner.run(test_mode=False)
        reward_scaler.fit(episode_batch)

    while runner.t_env <= args.t_max:

        # Step 1: Collect samples
        with torch.no_grad():
            episode_batch_1, episode_batch_2, train_stats, _, train_old_return = runner.run(test_mode=False)
            wandb_dict = {}
            wandb_dict.update({
                'train_return_old': train_old_return,
                'train_max_instock_sum': train_stats['max_in_stock_sum'],
                'train_mean_in_stock_sum': train_stats['mean_in_stock_sum']
            })
            for i in range(env_info["n_warehouses"]):
                wandb_dict.update({
                'train_mean_excess_sum_store_'+str(i+1): train_stats['mean_excess_sum_store_'+str(i+1)],
                'train_mean_backlog_sum_store_'+str(i+1): train_stats['mean_backlog_sum_store_'+str(i+1)],
                'train_mean_in_stock_sum_store_'+str(i+1): train_stats['mean_in_stock_sum_store_'+str(i+1)]
            })

            if args.use_reward_normalization:
                episode_batch = reward_scaler.transform(episode_batch)

            buffer_1.insert_episode_batch(episode_batch_1)
            buffer_2.insert_episode_batch(episode_batch_2)

        # Step 2: Train
        if buffer_1.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if (
                args.accumulated_episodes
                and next_episode % args.accumulated_episodes != 0
            ):
                continue

            episode_sample_1 = buffer_1.sample(args.batch_size)
            episode_sample_2 = buffer_2.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample_1.max_t_filled()
            episode_sample_1 = episode_sample_1[:, :max_ep_t]
            episode_sample_2 = episode_sample_2[:, :max_ep_t]

            if episode_sample_1.device != args.device:
                episode_sample_1.to(args.device)
                episode_sample_2.to(args.device)
            # TODO: 暂时注释掉
            # learner_1.train(episode_sample_1, runner.t_env, episode)
            learner_2.train(episode_sample_2, runner.t_env, episode)
            del episode_sample_1
            del episode_sample_2

            



        # Step 3: Evaluate
    
        if runner.t_env >= last_test_T + args.test_interval:
        # if True:

            # Log to console
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()
            last_test_T = runner.t_env

            # Evaluate the policy executed by argmax for the corresponding Q
            val_stats, val_lambda_return, val_old_return = \
                val_runner.run(test_mode=True, lbda_index=0)
            test_stats, test_lambda_return, test_old_return = \
                test_runner.run(test_mode=True, lbda_index=0)
            wandb_dict.update({
                'val_return_old': val_old_return,
                'val_max_instock_sum': val_stats['max_in_stock_sum'],
                'val_mean_in_stock_sum': val_stats['mean_in_stock_sum'],
                'test_return_old': test_old_return,
                'test_max_instock_sum': test_stats['max_in_stock_sum'],
                'test_mean_in_stock_sum': test_stats['mean_in_stock_sum'],
            })
            for i in range(env_info["n_warehouses"]):
                wandb_dict.update({
                'test_mean_excess_sum_store_'+str(i+1): test_stats['mean_excess_sum_store_'+str(i+1)],
                'test_mean_backlog_sum_store_'+str(i+1): test_stats['mean_backlog_sum_store_'+str(i+1)],
                'test_mean_in_stock_sum_store_'+str(i+1): test_stats['mean_in_stock_sum_store_'+str(i+1)],
                'val_mean_excess_sum_store_'+str(i+1): val_stats['mean_excess_sum_store_'+str(i+1)],
                'val_mean_backlog_sum_store_'+str(i+1): val_stats['mean_backlog_sum_store_'+str(i+1)],
                'val_mean_in_stock_sum_store_'+str(i+1): val_stats['mean_in_stock_sum_store_'+str(i+1)]
            })
                
            if val_old_return > val_best_return:
                val_best_return = val_old_return
                print("new best val result : {}".format(val_old_return))
                print("new test result : {}".format(test_old_return))
                save_path = os.path.join(
                args.local_results_path, args.unique_token, "models", str(runner.t_env)) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "models", str(runner.t_env))
                save_path = save_path.replace('*', '_')
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner_1.save_models(save_path, postfix="_1_best")
                learner_2.save_models(save_path, postfix="_2_best")


        if args.use_wandb:
            wandb.log(wandb_dict, step=runner.t_env)

        # Step 4: Save model
        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, args.unique_token, "models", str(runner.t_env)
            ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "models", str(runner.t_env))
            save_path = save_path.replace('*', '_')
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner_1.save_models(save_path, '_1_'+str(runner.t_env))
            learner_2.save_models(save_path, '_2_'+str(runner.t_env))

        # Step 5: Visualize
        if args.visualize and ((runner.t_env - visual_time) / args.visualize_interval >= 1.0):

            visual_time = runner.t_env
            visual_outputs_path = os.path.join(
                args.local_results_path, args.unique_token, "visual_outputs"
            )
            logger.console_logger.info(
                f"Saving visualizations to {visual_outputs_path}/{runner.t_env}"
            )

            # visual_runner.run_visualize(visual_outputs_path, runner.t_env)
            visual_runner.run_visualize(visual_outputs_path, runner.t_env)

        # Step 6: Finalize
        episode += args.batch_size_run
        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # Close the environments
    runner.close_env()
    val_runner.close_env()
    test_runner.close_env()
    logger.console_logger.info("Finished Training")


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