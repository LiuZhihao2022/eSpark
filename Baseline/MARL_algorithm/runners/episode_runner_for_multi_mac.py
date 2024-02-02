from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from Baseline.OR_algorithm.base_stock import *
import numpy as np


class EpisodeRunner4MultiMac:

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

    def setup(self, scheme, groups, preprocess, mac_list, set_stock_levels = None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac_list = mac_list
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
        self.batch_list = []
        for i in range(self.args.mac_num):
            self.batch_list.append(self.new_batch())
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        for i in range(self.args.mac_num):
            self.mac_list[i].init_hidden(batch_size=self.batch_size)

        while not terminated:
            actions_list = []
            for i in range(self.args.mac_num):
                pre_transition_data = {
                    "state": [self.env.get_state()[int(self.env.get_state_size()/self.args.mac_num)*i:int(self.env.get_state_size()/self.args.mac_num)*(i+1)]],
                    "avail_actions": [self.env.get_avail_actions()[int(len(self.env.get_avail_actions())/self.args.mac_num)*i:int(len(self.env.get_avail_actions())/self.args.mac_num)*(i+1)]],
                    "obs": [self.env.get_obs()[int(len(self.env.get_obs())/self.args.mac_num)*i:int(len(self.env.get_obs())/self.args.mac_num)*(i+1)]]
                }
                self.batch_list[i].update(pre_transition_data, ts=self.t)

                # Pass the entire batch of experiences up till now to the agents
                # Receive the actions for each agent at this timestep in a batch of size 1
                
                actions = self.mac_list[i].select_actions(self.batch_list[i], t_ep=self.t, t_env=self.t_env, test_mode=test_mode)[0]
                actions = actions.to("cpu").detach().numpy()
                actions_list.append(actions)
                # TODO:暂时到这里
            actions = np.concatenate(actions_list).reshape(-1)
            reward, terminated, env_info = self.env.step(actions)
            episode_return += reward

            for i in range(self.args.mac_num):
                post_transition_data= {
                    "actions": actions[i],
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }
                self.batch_list[i].update(post_transition_data, ts=self.t)

            self.t += 1
        for i in range(self.args.mac_num):
            last_data = {
                "state": [self.env.get_state()[int(self.env.get_state_size()/self.args.mac_num)*i:int(self.env.get_state_size()/self.args.mac_num)*(i+1)]],
                "avail_actions": [self.env.get_avail_actions()[int(len(self.env.get_avail_actions())/self.args.mac_num)*i:int(len(self.env.get_avail_actions())/self.args.mac_num)*(i+1)]],
                "obs": [self.env.get_obs()[int(len(self.env.get_obs())/self.args.mac_num)*i:int(len(self.env.get_obs())/self.args.mac_num)*(i+1)]]
            }
            self.batch_list[i].update(last_data, ts=self.t)

        # Select actions in the last stored state
        for i in range(self.args.mac_num):
            actions = self.mac_list[i].select_actions(self.batch_list[i], t_ep=self.t, t_env=self.t_env, test_mode=test_mode)[0]
            actions = actions.to("cpu").detach()
            self.batch_list[i].update({"actions": actions}, ts=self.t)

        return self.batch_list
    
    def run_visualize(self, visualize_path, t):
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
