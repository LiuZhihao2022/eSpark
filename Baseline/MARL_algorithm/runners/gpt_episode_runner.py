from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import time
from collections import defaultdict
class GptEpisodeRunner:

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

    def setup(self, scheme, groups, preprocess, mac, mask_func):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.mask_func = mask_func
    def update_mask_func(self, mask_func):
        self.mask_func = mask_func

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
        total_time = 0
        step_times = 0
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        total_mask_list = []
        mask_components_list = []
        while not terminated:
            
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            if self.args.mac == "mappo_mac":
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            elif self.args.mac == "dqn_mac" or self.args.mac == "ldqn_mac":
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, 
                    lbda_indices=None, test_mode=test_mode)
            
            reward, terminated, env_info = self.env.step(actions[0].reshape(-1))
            avail_actions = np.array(self.env.get_avail_actions())
            try:
            # 不用try- except了。现在runner里面只有一个mask函数，如果报错了外面的线程有try会接收到错误，接收到了就会返回错误信息并且关闭线程
                if self.mask_func is not None and not np.any(terminated):
                    total_mask, mask_components = self.mask_func(self.env._env.agent_states, self.env._env.supply_chain, self.env._env.config['action']['space'])
                    has_nan = np.any(np.isnan(total_mask))  
                    has_inf = np.any(np.isinf(total_mask))
                    if has_nan or has_inf:
                        raise ValueError("Total mask contains NaN or Inf values")
                    if np.any(np.all(total_mask == 0, axis = -1)):
                        raise ValueError("All actions of a certain warehouse are masked")
                    if not np.all(np.logical_or(total_mask == 0, total_mask == 1)):
                        raise ValueError("Elements of total mask can only be 0 or 1")
                    avail_actions = (avail_actions * total_mask.reshape(avail_actions.shape)).astype('int64').tolist()
            # 如果出现错误直接就返回错误的文本了。在主线程里通过判断返回值类型来确定是什么结果
            except Exception as e:
                return ("error", str(e))
            
            total_mask_list.append(total_mask)
            mask_components_list.append(mask_components)
            step_times += 1
            episode_return += reward

            post_transition_data = {
                "actions": actions[0].reshape(-1).to('cpu').detach(),
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
        

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [avail_actions],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if self.args.mac == "mappo_mac":
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        elif self.args.mac == "dqn_mac" or self.args.mac == "ldqn_mac":
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, 
                lbda_indices=None, test_mode=test_mode)
        self.batch.update({"actions": actions[0].reshape(-1).to('cpu').detach()}, ts=self.t)

        if not test_mode:
            self.t_env += self.t
        profit = self.env.get_profit()
        total_mask_distribution = np.array(total_mask_list).mean(axis = 0)
        mask_components_distribution = defaultdict(list)  
        for d in mask_components_list:  
            for k, v in d.items():  
                mask_components_distribution[k].append(v)  
        for k, v in mask_components_distribution.items():  
            mask_components_distribution[k] = np.mean(v, axis=0)  
        # 现在暂时用总的profit看看能不能跑通
        return self.batch, profit, total_mask_distribution, mask_components_distribution
    
    def run_visualize(self,visualize_path, t):
        self.reset()
        self.run(test_mode = True)
        self.env._env.visualizer.vis_path = visualize_path + '/' + str(t)
        self.env.render()
        # print("Total return : {}".format(np.sum(self.env.get_profit())))

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
