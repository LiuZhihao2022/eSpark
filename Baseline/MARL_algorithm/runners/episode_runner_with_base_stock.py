from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from Baseline.OR_algorithm.base_stock import *
import numpy as np


class EpisodeRunnerWithBaseStock:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        # For EpisodeRunner set batch_size default to 1
        # self.batch_size = self.args.batch_size_run
        self.batch_size = 1
        # assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, set_stock_levels = None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        if not isinstance(set_stock_levels, np.ndarray):
            self.set_stock_levels = get_multilevel_stock_level(self.env._env)
        else:
            self.set_stock_levels = set_stock_levels

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        replenish = self.set_stock_levels.copy()
        replenish = np.where(replenish >= 0, replenish, 0) / (self.env._env.get_demand_mean() + 0.00001)

        discrete_action = self.env._env.config['action']['space']
        actions = np.array([[np.argmin(np.abs(np.array(discrete_action) - a)) for a in row]  
            for row in replenish])
        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()[:int(self.env.get_state_size()/self.env.n_warehouses)]],
                "avail_actions": [self.env.get_avail_actions()[:int(len(self.env.get_avail_actions())/self.env.n_warehouses)]],
                "obs": [self.env.get_obs()[:int(len(self.env.get_obs())/self.env.n_warehouses)]]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            
            actions_from_rl = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            actions[0] = actions_from_rl[0].to("cpu").detach()
            reward, terminated, env_info = self.env.step(actions)
            episode_return += reward
            post_transition_data = {
                "actions": actions[0],
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()[:int(self.env.get_state_size()/self.env.n_warehouses)]],
            "avail_actions": [self.env.get_avail_actions()[:int(len(self.env.get_avail_actions())/self.env.n_warehouses)]],
            "obs": [self.env.get_obs()[:int(len(self.env.get_obs())/self.env.n_warehouses)]]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions_from_rl = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        actions[0] = actions_from_rl[0].to("cpu").detach()
        self.batch.update({"actions": actions[0]}, ts=self.t)

        return self.batch
    
    def run_visualize(self, visualize_path, t):
        # self.reset()

        # terminated = False
        # self.mac.init_hidden(batch_size=self.batch_size)

        # while not terminated:
        #     pre_transition_data = {
        #         "state": [self.env.get_state()[:int(self.env.get_state_size()/self.env.n_warehouses)]],
        #         "avail_actions": [self.env.get_avail_actions()[:int(len(self.env.get_avail_actions())/self.env.n_warehouses)]],
        #         "obs": [self.env.get_obs()[:int(len(self.env.get_obs())/self.env.n_warehouses)]]
        #     }
        #     self.batch.update(pre_transition_data, ts=self.t)
        #     if self.args.mac == "mappo_mac":
        #         action_from_rl = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True)
        #     elif self.args.mac == "dqn_mac" or self.args.mac == "ldqn_mac":
        #         action_from_rl = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, 
        #             lbda_indices=None, test_mode=True)
        #     # action_from_rl = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode = True)
        #     replenish = self.set_stock_levels - self.env._env.get_in_stock() - self.env._env.get_in_transit()
        #     replenish = np.where(replenish >= 0, replenish, 0) / (self.env._env.get_demand_mean() + 0.00001)

        #     discrete_action = self.env._env.config['action']['space']
        #     actions = np.array([[np.argmin(np.abs(np.array(discrete_action) - a)) for a in row]  
        #         for row in replenish])
        #     # RL algorithm with BSs. Top layer uses RL algorithm and others use BSs.
        #     actions[0] = action_from_rl[0].detach().cpu().numpy()[0]
        #     actions = actions.flatten()
        #     # Pass the entire batch of experiences up till now to the agents
        #     # Receive the actions for each agent at this timestep in a batch of size 1
        #     # actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode = True)
        #     reward, terminated, env_info = self.env.step(actions)
        #     self.t += 1

        # self.env._env.visualizer.vis_path = visualize_path + '/' + str(t)
        # self.env.render()


        self.reset()
        self.run(test_mode = True)
        self.env._env.visualizer.vis_path = visualize_path + '/' + str(t)
        self.env.render()
        print("Total return : {}".format(np.sum(self.env.get_profit())))

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()