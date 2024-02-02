import numpy as np
class SimpleMultiEchelonEnv(object):
    def __init__(self, 
    n_agents = 100,
        task_type = "Standard",
        mode = "train",
        time_limit=1460,
        vis_path=None,
        echelon_num = 5, 
        embedding_dim = 6, 
        sku_num = 1,
        seed = 101,
        beta = None,
        **kwargs):
    
        self.n_warehouses = echelon_num
        self.embedding_dim = embedding_dim
        self.sku_num = sku_num
        self.seed = seed
        self.episode_limit = time_limit
        np.random.seed(seed)
        # embedding改成与sku数目也有关
        embedding_matrix = np.random.randn(echelon_num*sku_num, embedding_dim)
        self.agent_embedding = np.eye(echelon_num*sku_num).dot(embedding_matrix)
        selling_profit_dict = {4:8, 5:30, 6:78, 7:150, 8:600}
        self.holding_cost = 5
        self.backlog = 5
        if beta is None:
            # self.selling_profit = 15*self.n_warehouses
            self.selling_profit = selling_profit_dict[self.n_warehouses]
        else:
            self.selling_profit = beta * (2**echelon_num)
            # self.selling_profit = 5000
            # self.selling_profit = 78
        # self.selling_profit = 150
    # 接受的all_actions是一个二维的，第一维是sku，第二维是warehouse
    def step(self, all_actions):
        """ Returns reward, terminated, info """
        # TODO: just for test  !! reverse actions
        # all_actions = 1 - all_actions
        all_actions = all_actions.reshape(self.sku_num, -1)
        self.env_t += 1
        total_balance = 0

        # actions = [int(a) for a in actions]
        for s in range(self.sku_num):
            actions = all_actions[s]
            in_stock = np.zeros(self.n_warehouses)
            balance = 0
            for i in range(len(actions)):
                if i == 0:
                    if actions[i] == 1:
                        in_stock[0] = 1

                elif i > 0 and i < len(actions) - 1:
                    if actions[i] == 1:
                        if in_stock[i-1] == 1:
                            in_stock[i-1] = 0
                            in_stock[i] = 1
                            # balance += self.selling_profit
                        elif in_stock[i-1] == 0:
                            balance -= self.backlog

                elif i == len(actions) - 1:
                    if actions[i] == 1:
                        if in_stock[i-1] == 1:
                            # 已经是最后一层了，in_stock[i]不需要再设置为1了。进货了也会立马买
                            in_stock[i-1] = 0
                            in_stock[i] = 1
                            # balance += self.selling_profit
                        elif in_stock[i-1] == 0:
                            balance -= self.backlog
                    if in_stock[i] == 1:
                        balance += self.selling_profit
                        in_stock[i] = 0
                    else:
                        # 已经是最后一层了，customer一定有需求，所以立马得到一个backlog
                        balance -= self.backlog
            balance -= self.holding_cost * in_stock.sum()
            total_balance += balance
        return total_balance, True, None

        

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.agent_embedding

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.agent_embedding[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.embedding_dim

    def get_state(self):
        return self.agent_embedding.reshape(-1)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_warehouses * self.sku_num *  self.embedding_dim

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_warehouses* self.sku_num):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        # only action 0 and 1
        return np.ones(2)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # only action 0 and 1
        return 2

    def reset(self):
        """ Returns initial observations and states"""
        self.env_t = 0
        # return obs and state
        return self.agent_embedding, self.agent_embedding.reshape(-1)
    
    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        return self.seed

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_warehouses*self.sku_num,
                    "episode_limit": self.episode_limit,
                    "n_warehouses": self.n_warehouses}
        return env_info
