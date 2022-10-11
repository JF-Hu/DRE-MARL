import  os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
import time
from env_utils.env_utils import normalize_reward


class CoopNavi_recorder():

    def __init__(self, agent_num):
        self.moment_ave_len = 40
        self.train_stage_rewards = []
        self.episode_rewards = []
        self.eval_episode_rewards = []
        self.agent_rewards = [[] for _ in range(agent_num)]
        self.x_axis = []
        self.eval_x_axis = []
        self.time_step = 0
        self.collide_num = [[] for _ in range(agent_num)]
        self.train_time = 0
        self.last_reward_n, self.single_step_reward = [0.0 for _ in range(agent_num)], [0.0 for _ in range(agent_num)]

        self.record_time_consume = False
        self.time_consume_list = []
        self.record_relative_ratio_var = False
        self.relative_ratio_var = []
        self.current_time_consume = 0
        self.start_train_time = 0

    def save_to_file(self, path):
        file_path = path + "/train_result"
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        with open(file_path+"/train_result.pkl", "wb") as f:
            pickle.dump(zip(self.episode_rewards, self.x_axis), f)
            f.close()
        with open(file_path+"/eval_result.pkl", "wb") as f:
            pickle.dump(zip(self.eval_episode_rewards, self.eval_x_axis), f)
            f.close()

    def single_reward_recoder_reset(self):
        for i in range(len(self.last_reward_n)):
            self.last_reward_n[i] = 0.0
            self.single_step_reward[i] = 0.0

    def single_reward_recoder(self, rewards, scale_rate, time_step_on_this_ep):
        self.store_agents_reward(rewards)
        if time_step_on_this_ep == 0 and (np.sum(self.single_step_reward) != 0 or np.sum(self.last_reward_n) != 0):
            self.single_reward_recoder_reset()
        for item in range(len(rewards)):
            self.single_step_reward[item] = normalize_reward(rewards[item] - self.last_reward_n[item], rate=scale_rate)
            self.last_reward_n[item] = rewards[item]

    def store_agents_reward(self, data):
        self.train_stage_rewards.append(data)

    def store_ep_data(self, x_axis):
        self.episode_rewards.append(np.sum(self.train_stage_rewards))
        if len(self.episode_rewards) > 1000:
            self.episode_rewards.pop(0)
            self.x_axis.pop(0)
        self.train_stage_rewards.clear()
        self.x_axis.append(x_axis)

    def store_eval_ep_data(self, data, x_axis):
        self.eval_episode_rewards.append(data)
        self.eval_x_axis.append(x_axis)

    def update_current_time_consume(self, start_time):
        self.current_time_consume = time.time() - start_time

    def record_start_time(self, start_time):
        self.start_train_time = start_time

    def plot_ep_result(self, save_fic_path, add_name=None):
        result = []
        for i in range(len(self.episode_rewards)):
            result.append(np.mean(self.episode_rewards[i : np.minimum(i+self.moment_ave_len, len(self.episode_rewards))]))
        plt.figure()
        plt.plot(self.x_axis, self.episode_rewards, label="episode_rewards")
        plt.plot(self.eval_x_axis, self.eval_episode_rewards, label="eval_episode_rewards")
        plt.legend()
        if not os.path.exists(save_fic_path):
            os.mkdir(save_fic_path)
        if not add_name:
            plt.savefig(save_fic_path+"/mean_ep_reward_{}.png".format(
                "%02d%2d%2d%2d" % (datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().minute, datetime.datetime.now().second)))
        else:
            plt.savefig(save_fic_path + "/mean_ep_reward_ep_{}_{}.png".format(add_name,
                "%02d%2d%2d%2d" % (
                datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().minute,
                datetime.datetime.now().second)))

    def get_mean_episode_rewards(self):
        return np.mean(self.episode_rewards[-self.moment_ave_len:])

    def get_mean_agents_rewards(self):
        result = []
        for i in range(len(self.agent_rewards)):
            result.append(np.mean(self.agent_rewards[i][-self.moment_ave_len:]))










