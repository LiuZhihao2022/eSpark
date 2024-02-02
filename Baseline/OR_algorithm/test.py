import random
import sys
import os
import numpy as np
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../..")
sys.path.insert(0, env_dir)
from ReplenishmentEnv import make_env

def sS_policy(env, S1, s1, S2, s2):
    env.reset()
    done = False
    sku_count = len(env.get_sku_list())
    total_reward = np.zeros((env.warehouse_count, sku_count))
    while not done:
        mean_demand = env.get_demand_mean()
        action = (env.get_in_stock() + env.get_in_transit()) / (mean_demand + 0.0001)
        action[0] = np.where(action[0] < s1, S1 - action[0], 0)
        action[1] = np.where(action[1] < s2, S2 - action[1], 0)
        state, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward, info["balance"]


def search_sS(env, search_range=np.arange(0, 12.1, 0.5)):
    env.reset()
    max_reward = 0
    best_S1, best_s1, best_S2, best_s2 = 0, 0, 0, 0
    
    for S1 in search_range:
        for s1 in np.arange(0, S1 + 0.1, 0.5):
            for S2 in np.arange(0.0, 12.1, 0.5): 
                for s2 in np.arange(0, S2 + 0.1, 0.5):
                    reward, _ = sS_policy(env, S1, s1, S2, s2)
                    reward = np.sum(reward)
                    if reward >= max_reward:
                        best_S1, best_s1, best_S2, best_s2 = S1, s1, S2, s2
                        max_reward = reward
                        print(S1, s1, S2, s2, max_reward)

    return best_S1, best_s1, best_S2, best_s2

output_dir = "output"
env_name = "different_order_cost"
vis_path = os.path.join(output_dir, env_name, "sS_hindsight")

env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode='test')
best_S1, best_s1, best_S2, best_s2 = search_sS(env_train)
print("Best")
print(best_S1, best_s1, best_S2, best_s2)

env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode='test', vis_path=vis_path)
_, balance = sS_policy(env_test, best_S1, best_s1, best_S2, best_s2)
print("Best balance : {}".format(balance))
env_test.render()