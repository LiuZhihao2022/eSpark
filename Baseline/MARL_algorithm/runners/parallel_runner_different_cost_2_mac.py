import os
import pdb
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pipe, Process

import numpy as np
import torch.nn.functional as F
import torch
import copy
import wandb
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from utils.timehelper import TimeStat
# sys.path.append(os.path.join(os.getcwd(), 'env/ReplenishmentEnv/OR_algorithm'))
# sys.path.append("../..")
# sys.path.append("../../env/ReplenishmentEnv/OR_algorithm/base_stock")
from Baseline.OR_algorithm.base_stock import *
# from ReplenishmentEnv.Baseline.OR_algorithm.base_stock import *
#from env.ReplenishmentEnv.Example.multilevel_base_stock import *
# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class Parallel4TwoMac:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]

        for i in range(len(env_args)):
            env_args[i]["seed"] += i

        self.ps = [
            Process(
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))),
            )
            for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        self.n_warehouses = self.env_info["n_warehouses"]
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_profits = []
        self.test_profits = []
        self.train_stats = {}
        self.test_stats = {}

        # self.time_stats = defaultdict(lambda: TimeStat(1000))
        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac_1, mac_2, set_stock_levels = None):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac_1 = mac_1
        self.mac_2 = mac_2
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        if not isinstance(set_stock_levels, np.ndarray):
            self.parent_conns[0].send(("get_stock_levels", None))
            self.set_stock_levels = self.parent_conns[0].recv()
        else:
            self.set_stock_levels = set_stock_levels


    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, test_mode=False, storage_capacity=None):

        self.batch_1 = self.new_batch()
        self.batch_2 = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("switch_mode", "eval" if test_mode else "train"))

        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
            
        pre_transition_data_1 = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "mean_action": [],
        }
        pre_transition_data_2 = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "mean_action": [],
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data_1["state"].append(data["state_1"])
            pre_transition_data_1["avail_actions"].append(data["avail_actions_1"])
            pre_transition_data_1["obs"].append(data["obs_1"])
            pre_transition_data_1["mean_action"].append(
                np.zeros([1, self.args.n_agents, self.args.n_actions])
            )
            pre_transition_data_2["state"].append(data["state_2"])
            pre_transition_data_2["avail_actions"].append(data["avail_actions_2"])
            pre_transition_data_2["obs"].append(data["obs_2"])
            pre_transition_data_2["mean_action"].append(
                np.zeros([1, self.args.n_agents, self.args.n_actions])
            )
        # TODO:这里怎么没有选择最前面的那个？而是一起给进去了？
        self.batch_1.update(pre_transition_data_1, ts=0)
        self.batch_2.update(pre_transition_data_2, ts=0)
        self.t = 0
        self.env_steps_this_run = 0

        self.train_returns = []
        self.test_returns = []
        self.train_profits = []
        self.test_profits = []

        if storage_capacity is not None:
            for parent_conn in self.parent_conns:
                parent_conn.send(("set_storage_capacity", storage_capacity))

    def run(self, lbda_index=None, test_mode=False, 
        visual_outputs_path=None, storage_capacity=None):

        self.reset(test_mode=test_mode, storage_capacity=storage_capacity)

        all_terminated = False
        episode_returns = np.zeros([self.batch_size, self.args.n_lambda])
        episode_lengths = [0 for _ in range(self.batch_size)]
        episode_balance = [0 for _ in range(self.batch_size)]
        if self.args.use_n_lambda:
            episode_individual_returns = np.zeros([self.batch_size, self.args.n_agents, self.args.n_lambda])
        else:
            episode_individual_returns = np.zeros([self.batch_size, self.args.n_agents])

        self.mac_1.init_hidden(batch_size=self.batch_size)
        self.mac_2.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = (
            []
        )  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        save_probs = getattr(self.args, "save_probs", False)

        while True:
            if self.args.mac == "mappo_mac":
                mac_output_1 = self.mac_1.select_actions(self.batch_1, t_ep=self.t, t_env=self.t_env, 
                    bs=envs_not_terminated, test_mode=test_mode)
                mac_output_2 = self.mac_2.select_actions(self.batch_2, t_ep=self.t, t_env=self.t_env, 
                    bs=envs_not_terminated, test_mode=test_mode)
            elif self.args.mac == "dqn_mac" or self.args.mac == "ldqn_mac":
                mac_output_1 = self.mac_1.select_actions(self.batch_1, t_ep=self.t, t_env=self.t_env, 
                    lbda_indices=None, bs=envs_not_terminated, test_mode=test_mode)
                mac_output_2 = self.mac_2.select_actions(self.batch_2, t_ep=self.t, t_env=self.t_env, 
                    lbda_indices=None, bs=envs_not_terminated, test_mode=test_mode)
            if save_probs:
                actions_1, probs_1 = mac_output_1
                actions_2, probs_2 = mac_output_2
            else:
                actions_1 = mac_output_1
                actions_2 = mac_output_2
            
            cpu_actions_1 = actions_1.to("cpu").numpy()
            cpu_actions_2 = actions_2.to("cpu").numpy()

            # Update the actions taken
            actions_chosen_1 = {
                "actions": actions_1.unsqueeze(1).to("cpu"),
            }
            actions_chosen_2 = {
                "actions": actions_2.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen_1["probs"] = probs_1.unsqueeze(1).to("cpu").detach()
                actions_chosen_2["probs"] = probs_2.unsqueeze(1).to("cpu").detach()

            self.batch_1.update(
                actions_chosen_1, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )
            self.batch_2.update(
                actions_chosen_2, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )
            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[
                        idx
                    ]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step_two_mac", (cpu_actions_1[action_idx], cpu_actions_2[action_idx])))
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data_1 = {
                "reward": [],
                "terminated": [],
                "individual_rewards": [],
                "cur_balance": []
            }
            post_transition_data_2 = {
                "reward": [],
                "terminated": [],
                "individual_rewards": [],
                "cur_balance": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data_1 = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "mean_action": [],
            }
            pre_transition_data_2 = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "mean_action": [],
            }


            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data_1["reward"].append((data["reward"],))
                    post_transition_data_2["reward"].append((data["reward"],))
                    post_transition_data_1["individual_rewards"].append(
                        data["info_1"]["individual_rewards"]
                    )
                    post_transition_data_2["individual_rewards"].append(
                        data["info_2"]["individual_rewards"]
                    )
                    post_transition_data_1["cur_balance"].append(
                        data["info_1"]["cur_balance"]
                    )
                    post_transition_data_2["cur_balance"].append(
                        data["info_2"]["cur_balance"]
                    )
                    episode_returns[idx] += data["reward"]
                    # TODO:这个似乎并没有用？
                    # if self.args.n_agents > 1:
                    #     episode_individual_returns[idx] += data["info"]["individual_rewards"]
                    # else:
                    #     episode_individual_returns[idx] += data["info"]["individual_rewards"][0]
                    # TODO:这个似乎并没有用？
                    # episode_balance[idx] = data["info"]["cur_balance"]

                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data_1["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data_1["state"].append(data["state_1"])
                    pre_transition_data_1["avail_actions"].append(data["avail_actions_1"])
                    pre_transition_data_1["obs"].append(data["obs_1"])
                    pre_transition_data_1["mean_action"].append(
                        # TODO:这里的n_actions是否少了或者多了
                        F.one_hot(actions_1[idx], self.env_info["n_actions"])
                        .float()
                        .mean(dim=0)
                        .view(1, 1, -1)
                        .repeat(1, self.args.n_agents, 1)
                        .cpu()
                        .numpy()
                    )

                    post_transition_data_2["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data_2["state"].append(data["state_2"])
                    pre_transition_data_2["avail_actions"].append(data["avail_actions_2"])
                    pre_transition_data_2["obs"].append(data["obs_2"])
                    pre_transition_data_2["mean_action"].append(
                        # TODO:这里的n_actions是否少了或者多了
                        F.one_hot(actions_2[idx], self.env_info["n_actions"])
                        .float()
                        .mean(dim=0)
                        .view(1, 1, -1)
                        .repeat(1, self.args.n_agents, 1)
                        .cpu()
                        .numpy()
                    )
            # Add post_transiton data into the batch_1
            self.batch_1.update(
                post_transition_data_1,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )
            self.batch_2.update(
                post_transition_data_2,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )
            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch_1.update(
                pre_transition_data_1, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )
            self.batch_2.update(
                pre_transition_data_2, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get profit for each env
        episode_profits = []
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_profit", None))
        for parent_conn in self.parent_conns:
            episode_profit = parent_conn.recv()
            episode_profits.append(episode_profit)

        # Get stats back for each env
        env_stats = []
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_profits = self.test_profits if test_mode else self.train_profits

        # log_prefix = "test_" if test_mode else ""
        if test_mode:
            log_prefix = "test" if visual_outputs_path is not None else "val"
        else:
            log_prefix = "train"
        if visual_outputs_path is not None:
            self.parent_conns[0].send(("visualize_render",visual_outputs_path))
            self.parent_conns[0].recv()
        infos = [cur_stats] + final_env_infos

        cur_stats.update(
            {
                k: sum(d.get(k, 0) for d in infos)
                for k in set.union(*[set(d) for d in infos])
            }
        )
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        max_in_stock_seq = [d['max_in_stock_sum'] for d in final_env_infos]
        cur_stats['max_in_stock_sum'] = np.mean(max_in_stock_seq)

        mean_in_stock_seq = [d['mean_in_stock_sum'] for d in final_env_infos]
        cur_stats['mean_in_stock_sum'] = np.mean(mean_in_stock_seq)

        for i in range(self.n_warehouses):
            mean_in_stock_store_seq = [d['mean_in_stock_sum_store_'+str(i+1)] for d in final_env_infos]
            mean_excess_store_seq = [d['mean_excess_sum_store_'+str(i+1)] for d in final_env_infos]
            mean_backlog_store_seq = [d['mean_backlog_sum_store_'+str(i+1)] for d in final_env_infos]
            cur_stats['mean_in_stock_sum_store_'+str(i+1)] = np.mean(mean_in_stock_store_seq)
            cur_stats['mean_excess_sum_store_'+str(i+1)] = np.mean(mean_excess_store_seq)
            cur_stats['mean_backlog_sum_store_'+str(i+1)] = np.mean(mean_backlog_store_seq)
        # TODO:这个return和profits有什么区别
        cur_returns.extend(episode_returns)
        cur_profits.extend(episode_profits)

        n_test_runs = (
            max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        )
        cur_profits = np.array(cur_profits)
        if test_mode:
            cur_returns = np.array(cur_returns)
            mean_returns = cur_returns.mean(axis=0)
            lambda_return = mean_returns[lbda_index]
            profits = (cur_profits.mean(axis=0)).sum(axis=-1)

            return cur_stats, lambda_return, profits
        else:
            cur_returns = np.array(cur_returns)
            mean_returns = cur_returns.mean(axis=0)
            lambda_return = mean_returns[lbda_index]
            profits = (cur_profits.mean(axis=0)).sum(axis=-1)
            return self.batch_1, self.batch_2, cur_stats, lambda_return, profits

    def get_overall_avg_balance(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_profit", None))
        cur_balances = []
        for parent_conn in self.parent_conns:
            cur_balances.append(parent_conn.recv())

        return np.mean(np.sum(np.array(cur_balances), axis=1))

    def _log(self, returns, individual_returns, profits, stats, prefix):
        self.logger.log_stat(prefix + "_return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "_return_std", np.std(returns), self.t_env)
        returns.clear()

        self.logger.log_stat(prefix + "_profit_mean", np.mean(profits), self.t_env)
        self.logger.log_stat(prefix + "_profit_std", np.std(profits), self.t_env)
        profits.clear()

        if self.args.use_wandb and self.args.n_agents <= 100:
            for i in range(self.args.n_agents):
                wandb.log(
                    {
                        f"SKUReturn/joint_{prefix}_sku{i+1}_mean": individual_returns[
                            :, i
                        ].mean()
                    },
                    step=self.t_env,
                )

            for i in range(self.args.n_agents):
                for parent_conn in self.parent_conns:
                    parent_conn.send(("get_reward_dict", None))
                reward_dicts = []
                for parent_conn in self.parent_conns:
                    reward_dicts.append(parent_conn.recv())

                for parent_conn in self.parent_conns:
                    parent_conn.send(("get_profit", None))
                cur_balances = []
                for parent_conn in self.parent_conns:
                    cur_balances.append(parent_conn.recv())
                wandb.log(
                    {
                        f"SKUReturn_{k}/joint_{prefix}_sku{i+1}_mean": np.mean(
                            [np.array(rd[k])[:, i].sum() / 1e6 for rd in reward_dicts]
                        )
                        for k in reward_dicts[0].keys()
                    },
                    step=self.t_env,
                )
                wandb.log(
                    {
                        f"SKUBalance/joint_{prefix}_sku{i+1}_mean": np.mean(
                            np.array(cur_balances)[:, i]
                        )
                    },
                    step=self.t_env,
                )
            wandb.log(
                    {
                        f"SumBalance/joint_{prefix}_sum": np.mean(
                            np.sum(np.array(cur_balances), 1)
                        )
                    },
                    step=self.t_env,
            )    

        if self.args.use_wandb:
            wandb.log(
                    {
                        f"instock_sum/{prefix}_max_in_stock_sum_mean": stats['max_in_stock_sum_mean'],
                        f"instock_sum/{prefix}_max_in_stock_sum_min": stats['max_in_stock_sum_min'],
                        f"instock_sum/{prefix}_max_in_stock_sum_max": stats['max_in_stock_sum_max'],
                    },
                    step=self.t_env,
            )    
        self.logger.log_stat(
            prefix + "_max_in_stock_sum_mean", stats['max_in_stock_sum_mean'], self.t_env
        )
        self.logger.log_stat(
            prefix + "_max_in_stock_sum_min", stats['max_in_stock_sum_min'], self.t_env
        )
        self.logger.log_stat(
            prefix + "_max_in_stock_sum_max", stats['max_in_stock_sum_max'], self.t_env
        )

        for k, v in stats.items():
            if k not in ["n_episodes", "individual_rewards"]:
                self.logger.log_stat(
                    prefix + "_" + k + "_mean", v / stats["n_episodes"], self.t_env
                )

        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step_two_mac":
            actions_1, actions_2 = data

            # replenish = stock_levels - env._env.get_in_stock() - env._env.get_in_transit()
            # TODO: temporarily modify to replenish 50 skus for top layer per day
            replenish = np.ones_like(env._env.get_in_stock())*50
            replenish = np.where(replenish >= 0, replenish, 0) / (env._env.get_demand_mean() + 0.00001)

            discrete_action = env._env.config['action']['space']
            actions = np.array([[np.argmin(np.abs(np.array(discrete_action) - a)) for a in row]  
                for row in replenish])
            # RL algorithm with BSs. Top layer uses RL algorithm and others use BSs.
            # actions[0] = action_from_rl
            actions[1] = actions_2
            actions = actions.flatten()


            # actions = np.concatenate((actions_1, actions_2))
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            env_info_1 = copy.deepcopy(env_info)
            env_info_2 = copy.deepcopy(env_info)
            # TODO:这个cur_balance是干嘛的
            env_info_1['cur_balance'] = env_info_1['cur_balance'].reshape(2, -1)[0]
            env_info_2['cur_balance'] = env_info_2['cur_balance'].reshape(2, -1)[1]
            env_info_1['individual_rewards'] = env_info_1['individual_rewards'].reshape(2, -1)[0]
            env_info_2['individual_rewards'] = env_info_2['individual_rewards'].reshape(2, -1)[1]
            # Return the observations, avail_actions and state to make the next action
            state_1 = env.get_state()[:int(env.get_state_size()/2)]
            state_2 = env.get_state()[int(env.get_state_size()/2):]
            avail_actions_1 = env.get_avail_actions()[:int(len(env.get_avail_actions())/2)]
            avail_actions_2 = env.get_avail_actions()[int(len(env.get_avail_actions())/2):]
            obs_1 = env.get_obs()[:int(len(env.get_obs())/2)]
            obs_2 = env.get_obs()[int(len(env.get_obs())/2):]
            remote.send(
                {
                    # Data for the next timestep needed to pick an action
                    "state_1": state_1,
                    "state_2": state_2,
                    "avail_actions_1": avail_actions_1,
                    "avail_actions_2": avail_actions_2,
                    "obs_1": obs_1,
                    "obs_2": obs_2,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info_1": env_info_1,
                    "info_2": env_info_2,
                    "info" : env_info
                }
            )
        elif cmd == "reset":
            env.reset()
            remote.send(
                {
                    "state_1": env.get_state()[:int(env.get_state_size()/2)],
                    "avail_actions_1": env.get_avail_actions()[:int(len(env.get_avail_actions())/2)],
                    "obs_1": env.get_obs()[:int(len(env.get_obs())/2)],
                    "state_2": env.get_state()[int(env.get_state_size()/2):],
                    "avail_actions_2": env.get_avail_actions()[int(len(env.get_avail_actions())/2):],
                    "obs_2": env.get_obs()[int(len(env.get_obs())/2):],
                }
            )
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "switch_mode":
            mode = data
            env.switch_mode(mode)
        elif cmd == "get_profit":
            # remote.send(env.get_profit().reshape(2, -1).sum(axis = 0))
            remote.send(env.get_profit())
        elif cmd == "get_reward_dict":
            remote.send(env._env.reward_monitor)
        elif cmd == "visualize_render":
            env.visualize_render(data)
            # profit = env.get_profit()
            # print("test_cur_avg_balances : {}".format(profit.sum()))
        elif cmd == "get_storage_capacity":
            remote.send(env._env.storage_capacity)
        elif cmd == "set_storage_capacity":
            env.set_storage_capacity(data)
        elif cmd == "get_stock_levels":
            stock_levels = get_multilevel_stock_level(env._env)
            remote.send(stock_levels)
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)