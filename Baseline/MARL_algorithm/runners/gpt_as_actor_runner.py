from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from test_gpt.my_gpt import MyGpt
import yaml
import json
import copy
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

    def setup(self, scheme, groups, preprocess, mac, gpt):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.GPT = gpt

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
    def build_input(self):
        date = "today"
        n_warehouses = self.env.n_warehouses
        sku_num = len(self.env._env.sku_list)
        input_format = {"1.number of warehouse" : n_warehouses, "2.number of SKU" : sku_num}
        warehouse_list = []
        # TODO:把这里补充完全
        for i in range(n_warehouses,0,-1):
            warehouse = "store{}".format(i)
            unit_storage_cost = self.env._env.supply_chain[warehouse, "unit_storage_cost"]
            volume = self.env._env.agent_states[warehouse, "volume", date, "all_skus"]
            basic_holding_cost = self.env._env.agent_states[warehouse, "basic_holding_cost", date, "all_skus"]
            holding_cost = basic_holding_cost + unit_storage_cost * volume
            if i != 1:
                demand = [None]*sku_num
            else:
                demand = self.env._env.agent_states[warehouse, "demand", date].tolist()
            warehouse_info = {
                "in stock" : self.env._env.get_in_stock(warehouse = warehouse).tolist(),
                "in transit" : self.env._env.get_in_transit(warehouse = warehouse).tolist(),
                "selling price" : self.env._env.agent_states[warehouse, "selling_price", date, "all_skus"].copy().tolist(),
                "procurement_cost" : self.env._env.agent_states[warehouse, "procurement_cost", date, "all_skus"].copy().tolist(),
                "capacity" : self.env._env.supply_chain[warehouse ,"capacity"],
                "unit_holding_cost" : holding_cost.item(),
                "backlog cost" : self.env._env.agent_states[warehouse, "backlog_ratio"].item(),
                "order cost" : self.env._env.agent_states[warehouse, "unit_order_cost"].item(),
                "demand" : demand,
                # "mean leading time in past 7 days" : self.env._env.get_average_vlt(warehouse = warehouse),
                "mean leading time in past 7 days" : int(np.average(self.env._env.agent_states[warehouse, "vlt", "lookback", "all_skus"],0)),
                "mean demand in past 7 days" : int(np.average(self.env._env.get_demand_mean(warehouse = warehouse),0)),
            }
            warehouse_list.append({"warehouse {}".format(i) : warehouse_info})
        input_format.update({"3.warehouses info":warehouse_list})
        return yaml.dump(input_format)

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        import logging  
  
        # 设置日志级别为WARNING，这将屏蔽掉DEBUG和INFO消息  
        openai_logger = logging.getLogger("openai") 
        openai_logger.setLevel(logging.WARNING) 
        logging.basicConfig(level=logging.WARNING)  
        see_last_n_transitions = 7
        interaction_messages = copy.deepcopy(self.GPT.init_messages)
        total_messages = copy.deepcopy(interaction_messages)
        # 以下是您的openapi相关代码  
        while not terminated:
            print("the {} step".format(self.t+1))
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            gpt_input = self.build_input()
            # only use the see_last_n_transitions
            if len(interaction_messages) > len(self.GPT.init_messages) + see_last_n_transitions*2:
                interaction_messages = interaction_messages[:len(self.GPT.init_messages)] + interaction_messages[-see_last_n_transitions*2:]
            interaction_messages, response = self.GPT.send_message(interaction_messages, gpt_input)
            total_messages.append({"role": "user", "content": gpt_input})
            total_messages.append({"role": "assistant", "content": response})
            self.GPT.save(total_messages)
            # TODO:check output format
            replenish = np.array(json.loads(response)['action'])
            replenish = np.where(replenish >= 0, replenish, 0) / (self.env._env.get_demand_mean() + 0.00001)
            discrete_action = self.env._env.config['action']['space']
            actions = np.array([[np.argmin(np.abs(np.array(discrete_action) - a)) for a in row]  
                for row in replenish]).reshape(-1)
            reward, terminated, env_info = self.env.step(np.array(actions.reshape(-1)))
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            
            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        # # directly use the last inpu as input
        # gpt_input = self.build_input()
        interaction_messages, response = self.GPT.send_message(interaction_messages, gpt_input)
        total_messages.append({"role": "user", "content": gpt_input})
        total_messages.append({"role": "assistant", "content": response})
        self.GPT.save(total_messages)
        replenish = np.array(json.loads(response)['action'])
        replenish = np.where(replenish >= 0, replenish, 0) / (self.env._env.get_demand_mean() + 0.00001)
        discrete_action = self.env._env.config['action']['space']
        actions = np.array([[np.argmin(np.abs(np.array(discrete_action) - a)) for a in row]  
            for row in replenish]).reshape(-1)
        self.batch.update({"actions": actions}, ts=self.t)

        if not test_mode:
            self.t_env += self.t

        return self.batch
    
    def run_visualize(self,visualize_path, t):
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
