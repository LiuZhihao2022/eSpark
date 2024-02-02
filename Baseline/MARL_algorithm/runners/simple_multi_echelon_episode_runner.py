from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch

class EpisodeRunner:

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
        self.optimal_time = 0
        self.make_part_action_to_one = False
        self.no_success_reward = False
        self.p = 0.1
        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    # def choose_mac(self, mac_list):


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
    
    def find_init_mac(self, mac_list):
        best_zero_ratio = -1
        best_mac_index = 0
        for i in range(len(mac_list)):
            self.reset()
            terminated = False
            episode_return = 0
            mac_list[i].random_init_hidden(batch_size=self.batch_size)
            pre_transition_data = {
                "state": [[self.env.get_state()]],
                "avail_actions": [[self.env.get_avail_actions()]],
                "obs": [[self.env.get_obs()]]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = mac_list[i].select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True)
            new_zero_ratio = torch.sum(actions[0] == 0).cpu().numpy().item()
            if new_zero_ratio > best_zero_ratio:
                best_zero_ratio = new_zero_ratio
                best_mac_index = i
        return mac_list[best_mac_index]
            


    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        # 已经挑选过的init_hidden了，就不用再init了
        # TODO:这里的和上面挑选的结果不一样？
        self.mac.init_hidden(batch_size=self.batch_size)
        # self.mac.random_init_hidden(batch_size=self.batch_size)
        while not terminated:

            pre_transition_data = {
                "state": [[self.env.get_state()]],
                "avail_actions": [[self.env.get_avail_actions()]],
                "obs": [[self.env.get_obs()]]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            action_to_save = actions[0].cpu().detach().numpy()
            prob_to_save = actions[1].cpu().detach().numpy()
            # 奇怪，进行下面的修改后，agent的动作会从全0重新回到一半一半？是因为无论选怎样的动作，得到的奖励都相同？可是buffer中是全1啊？
            if self.make_part_action_to_one:
                random_num = np.random.rand()
                if random_num < self.p:
                    action_to_save = np.ones_like(action_to_save)
                    # reward = np.ones_like(reward)*150/7
                    reward = 150
                    # action_log_probs = np.log(action_prob[:,:,1]).reshape(action_log_probs.shape)
                    
            if self.no_success_reward:
                reward = np.where(reward <0,reward,-25)
            print("reward : ",reward)
            print("success: ",reward)
            print("actions : ",action_to_save)
            print("prob : ",prob_to_save)
            post_transition_data = {
                "actions": action_to_save,
                # TODO:just for test
                "reward": [(reward/1e2,)],
                # "reward": [(reward,)],
                # "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "terminated": terminated,
                "probs": prob_to_save

            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [[self.env.get_state()]],
            "avail_actions": [[self.env.get_avail_actions()]],
            "obs": [[self.env.get_obs()]]
        }
        self.batch.update(last_data, ts=self.t)
        if reward > 0:
            self.optimal_time += 1
        # Select actions in the last stored state
        actions_last = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions_last[0].cpu().detach().numpy(), "probs": actions[1].cpu().detach().numpy()}, ts=self.t)
        

        # if self.t_env % 50 == 0:
        #     print(" training time : {}, reward : {}, optimal ratio till now : {}".format(self.t_env, reward, self.optimal_time/max(self.t_env, 1)))
        if not test_mode:
            self.t_env += self.t
        else:
            print("current agent policy : {}".format(actions[0].cpu().detach().numpy()))


        # 只有一次，所以return=reward
        return self.batch, reward
    
    def run_visualize(self,visualize_path, t):
        self.reset()

        terminated = False
        self.mac.init_hidden(batch_size=self.batch_size)
        pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
        self.batch.update(pre_transition_data,ts = self.t)
        while not terminated:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if self.args.mac == "mappo_mac":
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, 
                    test_mode=True)
            # TODO:QTRAN的动作选择还是有问题！
            elif self.args.mac == "dqn_mac" or self.args.mac == "ldqn_mac":
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, 
                    lbda_indices=None, test_mode=True)
            actions = actions[0].detach().cpu().numpy().flatten()
            reward, terminated, env_info = self.env.step(actions, test=True)
        self.env._env.visualizer.vis_path = visualize_path + '/' + str(t)
        self.env.render()

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
