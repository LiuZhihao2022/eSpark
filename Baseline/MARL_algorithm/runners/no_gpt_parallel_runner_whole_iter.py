from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import time
import os
from collections import defaultdict
from multiprocessing import Pipe, Process
from utils.timehelper import time_left, time_str
import copy
import torch.nn.functional as F


class NoGptParallelRunnerWholeIter:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.args.batch_size_run)]
        )
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.args.batch_size_run)]

        env = partial(env_fn, **self.args.env_args.copy())()
        self.env_info = env.get_env_info()
        self.episode_limit = self.env_info["episode_limit"]
        self.action_space = env._env.config['action']['space']

        self.episode_limit = self.env_info["episode_limit"]
        self.t = 0
        self.mask_func = None
        self.t_env = 0
        self.last_update_ep = 0
        self.last_evaluate_ep = 0
        self.train_returns = []
        self.test_returns = []
        self.train_profits = []
        self.test_profits = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, mask_func, learner, buffer, test_runner, val_runner, current_iter,
              subthread_id):
        if mask_func is not None:
            self.logger.console_logger.info(
                f"Error because exploration function of subthread {subthread_id} should be None!")
            return False, f"Error because exploration function of subthread {subthread_id} should be None! Exit thread {subthread_id}"
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.args.batch_size_run, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.mask_func = mask_func
        self.learner = learner
        self.buffer = buffer
        self.test_runner = test_runner
        self.val_runner = val_runner
        self.current_iter = current_iter
        self.subthread_id = subthread_id

        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.args.batch_size_run)]

        env = partial(env_fn, **self.args.env_args.copy())()
        self.env_info = env.get_env_info()
        self.episode_limit = self.env_info["episode_limit"]
        self.action_space = env._env.config['action']['space']

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
        return True, None

    def update_mask_func(self, mask_func):
        self.mask_func = mask_func
        if self.parent_conns is not None:
            for idx, parent_conn in enumerate(self.parent_conns):
                try:
                    parent_conn.send(("close", None))
                    parent_conn.close()
                    self.ps[idx].join()
                except OSError as e:
                    continue
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.args.batch_size_run)]
        )
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.args.batch_size_run)]

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
        self.t = 0

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            try:
                parent_conn.send(("close", None))
            except OSError as e:
                continue

    def reset(self, test_mode=False, storage_capacity=None):
        self.batch = self.new_batch()
        for parent_conn in self.parent_conns:
            parent_conn.send(("switch_mode", "eval" if test_mode else "train"))

        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "mean_action": [],
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["mean_action"].append(
                np.zeros([1, self.args.n_agents, self.args.n_actions])
            )

        self.batch.update(pre_transition_data, ts=0)
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
        episode_returns = np.zeros([self.args.batch_size_run, self.args.n_lambda])
        episode_lengths = [0 for _ in range(self.args.batch_size_run)]
        episode_balance = [0 for _ in range(self.args.batch_size_run)]
        if self.args.use_n_lambda:
            episode_individual_returns = np.zeros([self.args.batch_size_run, self.args.n_agents, self.args.n_lambda])
        else:
            episode_individual_returns = np.zeros([self.args.batch_size_run, self.args.n_agents])

        self.mac.init_hidden(batch_size=self.args.batch_size_run)
        terminated = [False for _ in range(self.args.batch_size_run)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = (
            []
        )  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        save_probs = getattr(self.args, "save_probs", False)
        err_list = []
        cpu_action_episode = []
        action_chosen_times = np.zeros((self.env_info["n_warehouses"], self.env_info["n_actions"]))
        # TODO:如果有一个env失败，那么整个runner都停止，不管其余线程是否失败
        while True:
            if all_terminated:
                break

            if self.args.mac == "mappo_mac":
                mac_output = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                     bs=envs_not_terminated, test_mode=test_mode)
            elif self.args.mac == "dqn_mac" or self.args.mac == "ldqn_mac":
                mac_output = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                     lbda_indices=None, bs=envs_not_terminated, test_mode=test_mode)

            if save_probs:
                actions, probs = mac_output
            else:
                actions = mac_output

            cpu_actions = actions.to("cpu").numpy()
            cpu_action_episode.append(cpu_actions)
            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }

            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu").detach()

            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )
            # for i in range(self.args.batch_size_run):
            #     # TODO:注意这里都是直接给的数值！(2,34)，后面需要替换为参数
            #     for j in range(self.env_info["n_warehouses"]):
            #         for k in range(int(self.env_info["n_agents"] / self.env_info["n_warehouses"])):
            #             action_chosen_times[j, cpu_actions[
            #                 i, int(self.env_info["n_agents"] / self.env_info["n_warehouses"]) * j + k]] += 1

            agents_per_warehouse = int(self.env_info["n_agents"] / self.env_info["n_warehouses"])
            for i in range(self.args.batch_size_run):
                # 对每个仓库处理其对应的代理所选择的动作
                for j in range(self.env_info["n_warehouses"]):
                    # 获取当前仓库的所有代理的行动选择
                    warehouse_agent_actions = cpu_actions[i, j * agents_per_warehouse:(j + 1) * agents_per_warehouse]

                    # 使用 NumPy 的 bincount 来计算每个行动被选择的次数，并累加到 action_chosen_times
                    # 注意：bincount 的长度是动作数量的最大值加一，所以我们需要先获取动作的最大值
                    max_action = warehouse_agent_actions.max()
                    action_counts = np.bincount(warehouse_agent_actions, minlength=max_action + 1)

                    # 累加当前批次的行动选择次数到总的 action_chosen_times 数组中
                    action_chosen_times[j, :len(action_counts)] += action_counts

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[
                        idx
                    ]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", (cpu_actions[action_idx])))
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": [],
                "individual_rewards": [],
                "cur_balance": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "mean_action": [],
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    recv = parent_conn.recv()
                    # TODO:如果有一个env失败，那么整个runner都停止，不管其余线程是否失败
                    if isinstance(recv, tuple) and recv[0] == 'error':
                        err = str(recv[1])
                        terminated = [True for i in range(self.args.batch_size_run)]
                        for idx, parent_conn in enumerate(self.parent_conns):
                            try:
                                parent_conn.send(("close", None))
                                parent_conn.close()
                                self.ps[idx].join()
                            except OSError as e:
                                continue
                        return False, err

                    data = recv
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))
                    post_transition_data["individual_rewards"].append(
                        data["info"]["individual_rewards"]
                    )
                    post_transition_data["cur_balance"].append(
                        data["info"]["cur_balance"]
                    )

                    episode_returns[idx] += data["reward"]

                    if self.args.n_agents > 1:
                        episode_individual_returns[idx] += data["info"]["individual_rewards"]
                    else:
                        episode_individual_returns[idx] += data["info"]["individual_rewards"][0]

                    episode_balance[idx] = data["info"]["cur_balance"]

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
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["mean_action"].append(
                        F.one_hot(actions[idx], self.env_info["n_actions"])
                        .float()
                        .mean(dim=0)
                        .view(1, 1, -1)
                        .repeat(1, self.args.n_agents, 1)
                        .cpu()
                        .numpy()
                    )

            # Add post_transiton data into the batch
            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        episode_profits = []
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_profit", None))
        for parent_conn in self.parent_conns:
            episode_profit = parent_conn.recv()
            episode_profits.append(episode_profit)


        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_profits = self.test_profits if test_mode else self.train_profits
        infos = [cur_stats] + final_env_infos

        cur_stats.update(
            {
                k: sum(d.get(k, 0) for d in infos)
                for k in set.union(*[set(d) for d in infos])
            }
        )
        # 统计只统计成功的batch些
        cur_stats["n_episodes"] = self.args.batch_size_run + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        max_in_stock_seq = [d['max_in_stock_sum'] for d in final_env_infos]
        cur_stats['max_in_stock_sum'] = np.mean(max_in_stock_seq)

        mean_in_stock_seq = [d['mean_in_stock_sum'] for d in final_env_infos]
        cur_stats['mean_in_stock_sum'] = np.mean(mean_in_stock_seq)
        cur_returns.extend(episode_returns)
        cur_profits.extend(episode_profits)

        profit = np.mean([np.sum(episode_profit) for episode_profit in episode_profits])
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

            return self.batch, cur_stats, profit, cpu_action_episode

    def run_visualize(self, visualize_path, t):
        self.test_runner.reset(visualize_path, t)
        # print("Total return : {}".format(np.sum(self.env.get_profit())))

    def run_iter(self):
        # start_time = time.time()
        action_chosen_times_list = []
        profit_list = []
        reward_component_stats = {"reward": [], "profit": [], "excess_cost": [], "order_cost": [], "holding_cost": [],
                                  "backlog_cost": []}
        test_reward_component_stats = {"reward": [], "profit": [], "excess_cost": [], "order_cost": [],
                                       "holding_cost": [], "backlog_cost": []}
        iter_interact_time = 0
        iter_train_time = 0
        iter_test_time = 0
        ep = 0
        cpu_action_list = []
        log_time = {"train_curve": [], "reward": [], "train": [], "evaluate": []}
        begin_time = time.time()
        while ep < self.args.iter_nepisode:
            if (ep + self.current_iter * self.args.iter_nepisode) % 100 == 0:
                self.logger.console_logger.info(f"Subthread : {self.subthread_id}, current episode : {ep}")

            # Step 1: Collect sample
            # TODO:这个run需要是并行的形式！返回值应该和gpt_parallel_runner相同，不过只用返回自己一个thread的
            interact_start_time = time.time()
            result = self.run()
            if len(result) == 2 and result[0] == False:
                self.logger.console_logger.info(f"thread {self.subthread_id} fails, stop training...")
                self.close_env()
                return result
            else:
                batch, cur_stats, profit, cpu_action_episode = result
            cpu_action_list.append(cpu_action_episode)
            self.buffer.insert_episode_batch(batch)
            profit_list.append(profit)
            interact_end_time = time.time()
            iter_interact_time += (interact_end_time - interact_start_time)
            ep += self.args.batch_size_run
            log_time["train"].append(ep + self.current_iter * self.args.iter_nepisode)
            if ep % 100 == 0:
                agents_per_warehouse = int(self.env_info["n_agents"]/self.env_info["n_warehouses"])
                cpu_action_list = np.array(cpu_action_list).reshape((-1, self.env_info["n_agents"]))
                action_chosen_times = np.zeros((self.env_info["n_warehouses"], self.env_info["n_actions"]))
                for j in range(self.env_info['n_warehouses']):
                    warehouse_agent_actions = cpu_action_list[:, j * agents_per_warehouse:(j + 1) * agents_per_warehouse].reshape(-1)
                    max_action = warehouse_agent_actions.max()
                    action_counts = np.bincount(warehouse_agent_actions, minlength=max_action + 1)
                    action_chosen_times[j, :len(action_counts)] += action_counts
                action_chosen_times_list.append(action_chosen_times)
                cpu_action_list = []

            # Step 2: Train
            if (self.args.accumulated_episodes is None) or (
                    self.current_iter * self.args.iter_nepisode + ep - self.last_update_ep >= self.args.accumulated_episodes):
                if self.buffer.can_sample(self.args.batch_size):
                    train_begin_time = time.time()
                    # print("learner train-------------------------")
                    self.last_update_ep = self.current_iter * self.args.iter_nepisode + ep
                    episode_sample = self.buffer.sample(self.args.batch_size)
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != self.args.device:
                        episode_sample.to(self.args.device)
                    # self.learner.train(episode_sample, self.t_env, episode, str(self.current_iter))
                    self.learner.train(episode_sample, self.t_env, self.current_iter * self.args.iter_nepisode + ep)
                    train_end_time = time.time()
                    iter_train_time += (train_end_time - train_begin_time)
                    log_time["train_curve"].append(ep + self.current_iter * self.args.iter_nepisode)

            # Step 3: Evaluate
            if self.args.iter_nepisode * self.current_iter + ep - self.last_evaluate_ep >= self.args.evaluate_nepisode and (
                    self.args.iter_nepisode * self.current_iter + ep) > 0:
                test_begin_time = time.time()
                self.last_evaluate_ep = self.args.iter_nepisode * self.current_iter + ep
                # time_remain = time_left_gpt(start_time, self.current_iter, ep, self.args.iter_nepisode, self.args.iters)
                batch_test, test_stats, test_old_return = \
                    self.test_runner.run(test_mode=True)
                batch_val, val_stats, val_old_return = \
                    self.val_runner.run(test_mode=True)
                for k in val_stats.keys():
                    test_reward_component_stats[k].append(test_stats[k])
                for k in val_stats.keys():
                    reward_component_stats[k].append(val_stats[k])
                test_end_time = time.time()
                iter_test_time += (test_end_time - test_begin_time)
                # TODO:这里的log_time填写完整后，继续在run中补充画图部分
                log_time["evaluate"].append(ep + self.current_iter * self.args.iter_nepisode)
                time_elapsed = time.time() - begin_time
                if ep != 0:
                    self.logger.console_logger.info(
                        f"Subthread id : {self.subthread_id}, time elapsed for iteration {self.current_iter}: {time_str(time_elapsed)}, time remain for this iteration : {time_str(max(self.args.iter_nepisode - ep, 0) / ep * time_elapsed)}. Interact time : {time_str(iter_interact_time)}, train time : {time_str(iter_train_time)}, test time : {time_str(iter_test_time)}")

        if len(cpu_action_list) > 0:
            agents_per_warehouse = int(self.env_info["n_agents"]/self.env_info["n_warehouses"])
            cpu_action_list = np.array(cpu_action_list).reshape((-1, self.env_info["n_agents"]))
            action_chosen_times = np.zeros((self.env_info["n_warehouses"], self.env_info["n_actions"]))
            for j in range(self.env_info['n_warehouses']):
                warehouse_agent_actions = cpu_action_list[:, j * agents_per_warehouse:(j + 1) * agents_per_warehouse].reshape(-1)
                max_action = warehouse_agent_actions.max()
                action_counts = np.bincount(warehouse_agent_actions, minlength=max_action + 1)
                action_chosen_times[j, :len(action_counts)] += action_counts
            action_chosen_times_list.append(action_chosen_times)
            cpu_action_list = []

        self.close_env()
        test_return_4_wandb = np.sum(test_reward_component_stats['reward'],
                                     axis=1).tolist()
        val_return_4_wandb = np.sum(reward_component_stats['reward'],
                                    axis=1).tolist()
        reward = np.array(reward_component_stats['reward'])
        eval_len = min(5, reward.shape[0])
        performance = reward[-eval_len:].sum() / eval_len
        training_curve_log = self.learner.training_curve_log.copy()
        self.learner.training_curve_log = []

        return True, (test_return_4_wandb, val_return_4_wandb, performance, training_curve_log, reward_component_stats,
                      test_reward_component_stats, profit_list, action_chosen_times_list, log_time)

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            try:
                actions = data
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