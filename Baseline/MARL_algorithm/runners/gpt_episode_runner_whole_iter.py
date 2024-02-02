from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import time
import os
from collections import defaultdict
from utils.timehelper import time_left, time_str
import copy
from components.episode_buffer import ReplayBuffer

class GptEpisodeRunnerWholeIter:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        assert self.args.batch_size_run == 1
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.env_info = self.env.get_env_info()
        self.episode_limit = self.env_info["episode_limit"]
        self.action_space = self.env._env.config['action']['space']

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
        if mask_func is None:
            self.logger.console_logger.info(
                                f"Error because exploration function of subthread {subthread_id} is None!")
            return False, f"exploration function does not implemented! Exit thread {subthread_id}"
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

        return True, None

    def update_mask_func(self, mask_func):
        self.mask_func = mask_func

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        self.env.close()

    def reset(self, test_mode=False, storage_capacity=None):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0 

    def run(self, lbda_index=None, test_mode=False,
            visual_outputs_path=None, storage_capacity=None):
        test_mode = False
        storage_capacity = None
        self.reset(test_mode=test_mode, storage_capacity=storage_capacity)
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.args.batch_size_run)
        total_mask_list = []
        mask_components_list = []
        # action_chosen_times = np.zeros((self.env_info["n_warehouses"], self.env_info["n_actions"]))
        cpu_action_episode = []
        while not terminated:
            avail_actions = np.array(self.env.get_avail_actions())
            self.batch.update(pre_transition_data, ts=self.t)
            # total_mask = avail_actions
            # mask_components = {}
            try:
                if not terminated:
                    total_mask, mask_components = self.mask_func(self.env._env.agent_states, self.env._env.supply_chain, self.env._env.config['action']['space'])
                    has_nan = np.any(np.isnan(total_mask))  
                    has_inf = np.any(np.isinf(total_mask))
                    if has_nan or has_inf:
                        raise ValueError("Total mask contains NaN or Inf values")
                    if not np.all(np.logical_or(total_mask == 0, total_mask == 1)):
                        raise ValueError("Elements of total mask can only be 0 or 1")
                    if np.any(np.all(total_mask == 0, axis=-1)):
                        raise ValueError("All actions of a certain warehouse are masked")
                    avail_actions = (avail_actions * total_mask.reshape(avail_actions.shape)).astype(
                        'int64').tolist()
            except Exception as e:
                return False, str(e)
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [avail_actions],
                "obs": [self.env.get_obs()]
            }
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if self.args.mac == "mappo_mac":
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            elif self.args.mac == "dqn_mac" or self.args.mac == "ldqn_mac":
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, 
                    lbda_indices=None, test_mode=test_mode)
            if self.args.save_probs:
                cpu_action = actions[0].reshape(-1).to('cpu').detach().numpy()
                prob = actions[1].reshape(-1).to('cpu').detach()
            else:
                cpu_action = actions.reshape(-1).to('cpu').detach().numpy()
            cpu_action_episode.append(cpu_action)

            # while not terminated:
            reward, terminated, env_info = self.env.step(cpu_action)
            # continue

            total_mask_list.append(total_mask)
            mask_components_list.append(mask_components)
            episode_return += reward


            post_transition_data = {
                "actions": cpu_action,
                "reward": [(reward,)],
                "terminated": [(terminated,)],
            }
            if self.args.save_probs:
                post_transition_data.update({"probs" : prob})

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1


        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if self.args.mac == "mappo_mac":
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        elif self.args.mac == "dqn_mac" or self.args.mac == "ldqn_mac":
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, 
                lbda_indices=None, test_mode=test_mode)
        
        if self.args.save_probs:
            cpu_action = actions[0].reshape(-1).to('cpu').detach()
            prob = actions[1].reshape(-1).to('cpu').detach()
        else:
            cpu_action = actions.reshape(-1).to('cpu').detach()
        self.batch.update({"actions": cpu_action}, ts=self.t)


        if not test_mode:
            self.t_env += self.t
        profit = self.env.get_profit()

        total_mask_distribution = np.array(total_mask_list).mean(axis=0)
        if len(mask_components_list[0]) > 0:
            mask_components_distribution = {
                k: np.mean([d.get(k, 0) for d in mask_components_list], axis=0)
                for k in set.union(*[set(d) for d in mask_components_list])
            }
        else:
            mask_components_distribution = None
        return self.batch, profit, total_mask_distribution, mask_components_distribution, cpu_action_episode


    def run_visualize(self, visualize_path, t):
        self.test_runner.reset(visualize_path, t)

    def run_iter(self):
        # start_time = time.time()
        total_mask_distribution_list = []
        mask_component_distribution_list = []
        profit_list = []
        reward_component_stats = {"reward": [], "profit": [], "excess_cost": [], "order_cost": [], "holding_cost": [],
                                  "backlog_cost": []}
        test_reward_component_stats = {"reward": [], "profit": [], "excess_cost": [], "order_cost": [],
                                       "holding_cost": [], "backlog_cost": []}
        iter_interact_time = 0
        iter_train_time = 0
        iter_test_time = 0
        ep = 0
        insert_buffer_time = 0
        cpu_action_list = []
        action_chosen_times_list = []
        log_time = {"train_curve": [], "reward": [], "train": [], "evaluate": []}
        begin_time = time.time()
        while ep < self.args.iter_nepisode:
            if (ep + self.current_iter * self.args.iter_nepisode) % 100 == 0:
                self.logger.console_logger.info(f"Subthread : {self.subthread_id}, current episode : {ep}")
            # Step 1: Collect sample
            # TODO:这个run需要是并行的形式！返回值应该和gpt_parallel_runner相同，不过只用返回自己一个thread的
            interact_start_time = time.time()                
            result = self.run()
            # self.run()
            if len(result) == 2 and result[0] == False:
                self.logger.console_logger.info(f"thread {self.subthread_id} fails, stop training...")
                self.close_env()
                return result
            else:
                batch, profit, total_mask_distribution, mask_components_distribution, cpu_action_episode= result
            


            cpu_action_list.append(cpu_action_episode)
            insert_buffer_time_begin = time.time()
            self.buffer.insert_episode_batch(batch)
            insert_buffer_time_end = time.time()
            insert_buffer_time += insert_buffer_time_end - insert_buffer_time_begin
            total_mask_distribution_list.append(total_mask_distribution)
            mask_component_distribution_list.append(mask_components_distribution)
            profit_list.append(profit)
            interact_end_time = time.time()
            iter_interact_time += (interact_end_time - interact_start_time)
            log_time["train"].append(ep + self.current_iter * self.args.iter_nepisode)

            if ep % 500 == 0:
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
            ep += self.args.batch_size_run

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
                      test_reward_component_stats, profit_list, total_mask_distribution_list,
                      mask_component_distribution_list, action_chosen_times_list, log_time)

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()

