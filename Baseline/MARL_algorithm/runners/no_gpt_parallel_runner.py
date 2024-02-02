import os
import pdb
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pipe, Process
import importlib
import numpy as np
import torch.nn.functional as F
import torch
import copy
import wandb
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from utils.timehelper import TimeStat

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class NoGptParallelRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.gpt_sample_num = self.args.gpt_sample_num
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
        # TODO:想到一开始怎么给到env_info后就可以吧init里的多线程删除了
        # self.parent_conns, self.worker_conns = zip(
        #     *[Pipe() for _ in range(self.gpt_sample_num)]
        # )
        env_fn = env_REGISTRY[self.args.env]
        self.parent_conns = None
        # env_args = [self.args.env_args.copy() for _ in range(self.gpt_sample_num)]

        # for i in range(len(env_args)):
        #     env_args[i]["seed"] += i

        # self.ps = [
        #     Process(
        #         target=env_worker,
        #         args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))),
        #     )
        #     for env_arg, worker_conn in zip(env_args, self.worker_conns)
        # ]
        # for p in self.ps:
        #     p.daemon = True
        #     p.start()
        # self.parent_conns[0].send(("get_env_info", None))
        # self.env_info = self.parent_conns[0].recv()
        env = partial(env_fn, **self.args.env_args.copy())()
        self.env_info = env.get_env_info()
        self.action_space = env._env.config['action']['space']
        self.episode_limit = self.env_info["episode_limit"]
        self.mask_func_list = None

    def setup(self, scheme, groups, preprocess, mac_list):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            1, # batch_size设置为1，每个mac用一个batch
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac_list = mac_list
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        
    def update_mask_and_succ_list(self, mask_func_list, successes):

        self.mask_func_list = mask_func_list
        self.successes_bool = successes
        self.successes_index = [index for index, value in enumerate(self.successes_bool) if value]  

        # 因为原来的线程接受不了新函数，所以直接开新线程吧
        # Make subprocesses for the envs
        if self.parent_conns is not None:
            for idx, parent_conn in enumerate(self.parent_conns):
                try:
                    parent_conn.send(("close", None))
                    parent_conn.close()
                    self.ps[idx].join()
                except OSError as e:
                    continue

        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.gpt_sample_num)]
        )
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.gpt_sample_num)]

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


    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, test_mode=False, storage_capacity=None):

        self.batch_list = [self.new_batch() for i in range(self.gpt_sample_num)]
        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("switch_mode", "eval" if test_mode else "train"))

        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
            
        # Get the obs, state and avail_actions back
        for idx, parent_conn in enumerate(self.parent_conns):
            
            data = parent_conn.recv()
            pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "mean_action": [],
            }
            pre_transition_data["state"].append(data["state"])
            # TODO:在第一步reset的时候要不要加gpt mask？
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["mean_action"].append(
                np.zeros([1, self.args.n_agents, self.args.n_actions])
            )
            self.batch_list[idx].update(pre_transition_data, ts=0)

        # self.batch.update(pre_transition_data, ts=0)

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
        episode_returns = np.zeros([self.gpt_sample_num, self.args.n_lambda])
        episode_lengths = [0 for _ in range(self.gpt_sample_num)]
        if self.args.use_n_lambda:
            episode_individual_returns = np.zeros([self.gpt_sample_num, self.args.n_agents, self.args.n_lambda])
        else:
            episode_individual_returns = np.zeros([self.gpt_sample_num, self.args.n_agents])

        # self.mac.init_hidden(batch_size=int(sum(self.successes_bool)))
        # batch_size暂时就设置为1
        for mac in self.mac_list:
            mac.init_hidden(batch_size=1)
        # 一开始就要将fail的设置为terminated
        terminated = [not x for x in self.successes_bool]
        final_env_infos = (
            []
        )  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        save_probs = getattr(self.args, "save_probs", False)
        err_list = []
        # TODO:暂时这里就按照warehouse=2，action=34来手动设计
        action_chosen_times_list = [np.zeros((self.env_info["n_warehouses"], self.env_info["n_actions"])) for i in range(self.gpt_sample_num)]
        # 第一步时肯定不会全部失败，因为如果全部失败就不会进入run函数
        while True:
            # hidden的维度动态处理  TODO:那这里改变了一次岂不是就变了？不能再通过success_index索引了！
            # self.mac.hidden_states = self.mac.hidden_states[self.successes_index]
            # hidden state不变，但是通过envs_not_terminated来选择哪些动作是可以用的
            all_terminated = all(terminated)
            if all_terminated:
                break

            envs_not_terminated_before_action_select = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            # output_list只包含没有terminate的线程的输出
            output_list = []
            if self.args.mac == "mappo_mac":
                # mac_output = self.mac.select_actions(self.batch[self.successes_index], t_ep=self.t, t_env=self.t_env, 
                #     bs=envs_not_terminated_before_action_select, test_mode=test_mode)
                # 需要记录选择动作的时候每个动作对应的线程序号是多少。最后若线程减少了，可以通过剩余的线程序号去检索当时选择的动作
                for idx, mac in enumerate(self.mac_list):
                    if not terminated[idx]:
                        output = self.mac_list[idx].select_actions(self.batch_list[idx], t_ep=self.t, t_env=self.t_env, 
                            test_mode=test_mode)
                        output_list.append(output)
            elif self.args.mac == "dqn_mac" or self.args.mac == "ldqn_mac":
                # mac_output = self.mac.select_actions(self.batch[self.successes_index], t_ep=self.t, t_env=self.t_env, 
                #     lbda_indices=None, bs=envs_not_terminated_before_action_select, test_mode=test_mode)
                for idx, mac in enumerate(self.mac_list):
                    if not terminated[idx]:
                        output = self.mac_list[idx].select_actions(self.batch_list[idx], t_ep=self.t, t_env=self.t_env, 
                            lbda_indices=None, test_mode=test_mode)
                        output_list.append(output)
            else:
                raise NotImplementedError
            
            if save_probs:
                actions_list = [output[0] for output in output_list]
                probs_list = [output[1] for output in output_list]
                actions = torch.cat(actions_list, dim=0)
                probs = torch.cat(probs_list, dim=0)
            else:
                actions = torch.cat(output_list, dim=0)

            cpu_actions = actions.to("cpu").numpy()
            if  save_probs:
                actions_chosen_list = [{
                    "actions": a.unsqueeze(1).to("cpu"), "probs": p.unsqueeze(1).to("cpu").detach()
                } for a,p in zip(actions, probs)]
            else:
                actions_chosen_list = [{
                    "actions": a.unsqueeze(1).to("cpu"), 
                } for a in actions]
            # Update the actions taken
            # actions_chosen = {
            #     "actions": actions.unsqueeze(1).to("cpu"),
            # }

            # TODO:包含prob的情况以后再写吧
            # if save_probs:
            #     actions_chosen["probs"] = probs.unsqueeze(1).to("cpu").detach()

            # 只有没出问题的进行一个更新
            for i, idx in enumerate(envs_not_terminated_before_action_select):
                self.batch_list[idx].update(
                    actions_chosen_list[i], ts=self.t, mark_filled=False
                )
                # TODO:注意这里都是直接给的数值！(2,34)，后面需要替换为参数
                for j in range(self.env_info["n_warehouses"]):
                    for k in range(int(self.env_info["n_agents"]/self.env_info["n_warehouses"])):
                        action_chosen_times_list[idx][j, cpu_actions[i, int(self.env_info["n_agents"]/self.env_info["n_warehouses"])*j+k]] += 1


            # 注意：在程序外部判断用哪些batch更新，而不是在函数里面用bs参数来指定
            # self.batch[self.successes_index].update(
            #     actions_chosen, ts=self.t, mark_filled=False
            # )
            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated_before_action_select:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        mask_func = None if self.mask_func_list is None else self.mask_func_list[idx]
                        parent_conn.send(("step", (cpu_actions[action_idx], mask_func)))
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated_before_action_select before break
            # envs_not_terminated_before_action_select = [
            #     b_idx for b_idx, termed in enumerate(terminated) if not termed
            # ]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data_list = [{
                "reward": [],
                "terminated": [],
                "individual_rewards": [],
                "cur_balance": []
            } for i in range(self.gpt_sample_num)]
            # Data for the next step we will insert in order to select an action
            pre_transition_data_list = [{
                "state": [],
                "avail_actions": [],
                "obs": [],
                "mean_action": [],
            } for i in range(self.gpt_sample_num)]

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    recv = parent_conn.recv()
                    data = recv
                    # Remaining data for this current timestep
                    post_transition_data_list[idx]["reward"].append((data["reward"],))
                    post_transition_data_list[idx]["individual_rewards"].append(
                        data["info"]["individual_rewards"]
                    )
                    post_transition_data_list[idx]["cur_balance"].append(
                        data["info"]["cur_balance"]
                    )

                    episode_returns[idx] += data["reward"]

                    if self.args.n_agents > 1:
                        episode_individual_returns[idx] += data["info"]["individual_rewards"]
                    else:
                        episode_individual_returns[idx] += data["info"]["individual_rewards"][0]


                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    # 如果线程中断了，上面就停了，也不会将final info加入在内
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = np.any(data["terminated"])
                    post_transition_data_list[idx]["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data_list[idx]["state"].append(data["state"])
                    pre_transition_data_list[idx]["avail_actions"].append(data["avail_actions"])
                    pre_transition_data_list[idx]["obs"].append(data["obs"])
                    pre_transition_data_list[idx]["mean_action"].append(
                        F.one_hot(actions[envs_not_terminated_before_action_select.index(idx)], self.env_info["n_actions"])
                        .float()
                        .mean(dim=0)
                        .view(1, 1, -1)
                        .repeat(1, self.args.n_agents, 1)
                        .cpu()
                        .numpy()
                    )
            
            # Add post_transiton data into the batch
            # 不应该用self.successes_bool索引，不然会索引4个0
            self.successes_index = [index for index, value in enumerate(self.successes_bool) if value]  
            # TODO:如果全部线程都失败了，那么应该直接退。暂时先按照原来的输出格式输出。需要在run中补充如果全部失败了应该怎么办的代码
            if len(self.successes_index) == 0:
                for idx, parent_conn in enumerate(self.parent_conns):
                    try:
                        parent_conn.send(("close", None))
                        parent_conn.close()
                        self.ps[idx].join()
                    except OSError as e:
                        continue
                return ["\n".join(err_list)]
            # 只更新没有terminate的batch
            for idx in self.successes_index:
                self.batch_list[idx].update(
                    post_transition_data_list[idx],
                    # bs=envs_not_terminated,
                    ts=self.t,
                    mark_filled=False,
                )
            # self.batch[self.successes_index].update(
            #     post_transition_data,
            #     ts=self.t,
            #     mark_filled=False,
            # )

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            for idx in self.successes_index:
                self.batch_list[idx].update(
                    pre_transition_data_list[idx], 
                    # bs=envs_not_terminated, 
                    ts=self.t, 
                    mark_filled=True
                )

            # self.batch[self.successes_index].update(
            #     pre_transition_data, ts=self.t, mark_filled=True
            # )

        if not test_mode:
            self.t_env += self.env_steps_this_run/sum(self.successes_bool)

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

        cur_stats = self.test_stats if test_mode else self.train_stats #TODO: 为什么self.train_stats会不为空？根本没有赋值啊
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
        # 统计只统计成功的batch些
        cur_stats["n_episodes"] = len(self.successes_index) + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        max_in_stock_seq = [d['max_in_stock_sum'] for d in final_env_infos]
        cur_stats['max_in_stock_sum'] = np.mean(max_in_stock_seq)

        mean_in_stock_seq = [d['mean_in_stock_sum'] for d in final_env_infos]
        cur_stats['mean_in_stock_sum'] = np.mean(mean_in_stock_seq)

        # 统计只统计成功的batch些
        cur_returns.extend([episode_profits[i] for i in self.successes_index])
        cur_profits.extend([episode_profits[i] for i in self.successes_index])
        profit_list = [np.sum(episode_profit) for episode_profit in episode_profits]
        # 这句好像没用，所以没改
        n_test_runs = (
            max(1, self.args.test_nepisode // self.gpt_sample_num) * self.gpt_sample_num
        )

        if test_mode:
            cur_returns = np.array(cur_returns)
            mean_returns = cur_returns.mean(axis=0)
            lambda_return = mean_returns[lbda_index]

            cur_profits = np.array(cur_profits)
            profits = (cur_profits.mean(axis=0)).sum(axis=-1)

            return cur_stats, lambda_return, profits
        else:
            cur_returns = np.array(cur_returns)
            mean_returns = cur_returns.mean(axis=0)
            lambda_return = mean_returns[lbda_index]

            cur_profits = np.array(cur_profits)
            profits = (cur_profits.mean(axis=0)).sum(axis=-1)
            
            return self.batch_list, self.successes_bool, cur_stats, profit_list, action_chosen_times_list

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
        if cmd == "step":
            try:
                actions, mask_func = data
                # Take a step in the environment
                reward, terminated, env_info = env.step(actions)
                # Return the observations, avail_actions and state to make the next action
                state = env.get_state()
                avail_actions = np.array(env.get_avail_actions())
                obs = env.get_obs()
                remote.send((
                    {
                        # Data for the next timestep needed to pick an action
                        "state": state,
                        "avail_actions": avail_actions,
                        "obs": obs,
                        # Rest of the data for the current timestep
                        "reward": reward,
                        "terminated": terminated,
                        "info": env_info,
                    })
                )
            except Exception as e:
                remote.send(("error", str(e)))
                # 不close，因为线程虽然报错了，但是后面更新mask函数后还可能接着要用，所以不要close
                # remote.close()
        elif cmd == "reset":
            env.reset()
            remote.send(
                {
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs(),
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