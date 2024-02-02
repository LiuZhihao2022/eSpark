from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import time
import os
from collections import defaultdict
from utils.timehelper import time_left, time_str
class GptEpisodeRunnerWholeIter:

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

    def setup(self, scheme, groups, preprocess, mac, mask_func, learner, buffer, test_runner, current_iter, mask_output_dir, subthread_id):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.mask_func = mask_func
        self.learner = learner
        self.buffer = buffer
        self.test_runner = test_runner
        self.current_iter = current_iter
        self.mask_output_dir = mask_output_dir
        self.subthread_id = subthread_id
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
                return [str(e)]
            
            total_mask_list.append(total_mask)
            mask_components_list.append(mask_components)
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
        mask_components_distribution = {
                k: np.mean([d.get(k, 0) for d in mask_components_list], axis=0)
                for k in set.union(*[set(d) for d in mask_components_list])
                }
        # mask_components_distribution = defaultdict(list)  
        # for d in mask_components_list:  
        #     for k, v in d.items():  
        #         mask_components_distribution[k].append(v)  
        # for k, v in mask_components_distribution.items():  
        #     mask_components_distribution[k] = np.mean(v, axis=0)  
        # 现在暂时用总的profit看看能不能跑通
        return self.batch, profit, total_mask_distribution, mask_components_distribution
    
    def run_visualize(self,visualize_path, t):
        self.reset()
        self.run(test_mode = True)
        self.env._env.visualizer.vis_path = visualize_path + '/' + str(t)
        self.env.render()
        # print("Total return : {}".format(np.sum(self.env.get_profit())))
    def run_iter(self):
        start_time = time.time()
        total_mask_distribution_list = []
        mask_component_distribution_list = []
        reward_component_stats = {"reward" : [], "profit" : [], "excess_cost" : [], "order_cost" : [], "holding_cost" : [], "backlog_cost" : []}
        episode = 0
        last_test_T = 0
        while self.t_env < self.args.t_max:
            # Step 1: Collect sample
            result = self.run()
            if len(result) == 1:
                self.close_env()
                return False, result[0]
            else:
                batch, profit, total_mask_distribution, mask_components_distribution = result
            self.buffer.insert_episode_batch(batch)
            total_mask_distribution_list.append(total_mask_distribution)
            mask_component_distribution_list.append(mask_components_distribution)
            # Step 2: Train

            if self.buffer.can_sample(self.args.batch_size):
                next_episode = episode + self.args.batch_size_run
                if (self.args.accumulated_episodes == None) or (
                self.args.accumulated_episodes
                and next_episode % self.args.accumulated_episodes == 0
                ):
                    episode_sample = self.buffer.sample(self.args.batch_size)
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != self.args.device:
                        episode_sample.to(self.args.device)

                    self.learner.train(episode_sample, self.t_env, episode, str(self.current_iter))
            # Step 3: Evaluate
            if self.t_env >= last_test_T + self.args.test_interval:
                print("test-------------------------")
                # Log to console
                self.logger.console_logger.info(
                    "subthread_id: {}, t_env: {} / {}".format(self.subthread_id, self.t_env, self.args.t_max)
                )
                self.logger.console_logger.info(
                    "subthread_id: {}, Estimated time left: {}. Time passed: {}".format(
                        self.subthread_id, time_left(last_time, last_test_T, self.t_env, self.args.t_max),
                        time_str(time.time() - start_time),
                    )
                )
                last_time = time.time()
                last_test_T = self.t_env

                # Evaluate the policy executed by argmax for the corresponding Q
                # val_stats, val_lambda_return, val_old_return = \
                #     val_runner.run(test_mode=True, lbda_index=0)
                # val_stats, val_stats, val_old_return = \
                #     self.val_runner.run(test_mode=True)
                batch_test, test_stats, test_old_return = \
                    self.test_runner.run(test_mode=True)
                for k in test_stats.keys():
                    reward_component_stats[k].append(test_stats[k])
                # wandb_dict.update({
                #     'val_return_old': val_old_return,
                #     'val_max_instock_sum': val_stats['max_in_stock_sum'],
                #     'val_mean_in_stock_sum': val_stats['mean_in_stock_sum'],
                #     'test_return_old': test_old_return,
                #     'test_max_instock_sum': test_stats['max_in_stock_sum'],
                #     'test_mean_in_stock_sum': test_stats['mean_in_stock_sum'],
                # })

                # TODO:太多线程，不知道哪个是最好的模型，暂时先不在每个iter中进行保存了
                # if val_old_return > val_best_return:
                #     val_best_return = val_old_return
                #     print("new best val result : {}".format(val_old_return))
                #     print("new test result : {}".format(test_old_return))
                #     save_path = os.path.join(
                #     self.args.local_results_path, self.args.unique_token, "models", str(self.t_env)) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", self.args.unique_token, "models", str(self.t_env))
                #     save_path = save_path.replace('*', '_')
                #     os.makedirs(save_path, exist_ok=True)
                #     self.logger.console_logger.info("Saving models to {}".format(save_path))

                #     # learner should handle saving/loading -- delegate actor save/load to mac,
                #     # use appropriate filenames to do critics, optimizer states
                #     print("Update best model, val return : {}, test return : {}".format(val_old_return, test_old_return))
                #     self.learner.save_models(save_path, postfix="_best")

            # if args.use_wandb:
            #     wandb.log(wandb_dict, step=runner.t_env)

            # TODO: 模型保存也是在main thread中统一进行保存
            # Step 4: Save model
            # if self.args.save_model and (
            #     self.t_env - model_save_time >= self.args.save_model_interval
            #     or model_save_time == 0
            # ):
            #     model_save_time = self.t_env
            #     save_path = os.path.join(
            #         self.args.local_results_path, self.args.unique_token, "models", str(self.t_env)
            #     ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", self.args.unique_token, "models", str(self.t_env))
            #     save_path = save_path.replace('*', '_')
            #     os.makedirs(save_path, exist_ok=True)
            #     # logger.console_logger.info("Saving models to {}".format(save_path))

            #     # learner should handle saving/loading -- delegate actor save/load to mac,
            #     # use appropriate filenames to do critics, optimizer states
            #     self.learner.save_models(save_path, '_'+str(self.t_env))
            
            episode += 1
        self.close_env()
        return True, (reward_component_stats, total_mask_distribution_list, mask_component_distribution_list)
            



    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
