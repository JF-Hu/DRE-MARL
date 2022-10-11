
import copy
import random
import numpy as np
from buffer.trajectory_buffer import joint_trajectory_buffer

def calculate_TD_discount_reward(TD_reward_trajectory, gamma):
    discounted_reward = 0
    for item in TD_reward_trajectory[::-1]:
        discounted_reward = gamma * discounted_reward + item
    return discounted_reward

class bese_replay_buffer():
    def __init__(self, argus):
        self.argus = argus
        self.traj_buffer = joint_trajectory_buffer(agent_num=argus.env.agent_num)
        self.buffer_s, self.action_index, self.action_prob, self.action_dist, self.reward, self.next_buffer_s, self.next_action_index, self.terminate = [], [], [], [], [], [], [], []
        self.buffer_js, self.joint_action_index, self.joint_action_prob, self.joint_action_dist, self.joint_reward, self.next_buffer_js, self.next_joint_action_index, self.joint_terminate = [], [], [], [], [], [], [], []
        for index in range(argus.env.agent_num):
            self.buffer_s.append([])
            self.next_buffer_s.append([])
            self.reward.append([])
            self.action_index.append([])
            self.action_prob.append([])
            self.action_dist.append([])
            self.next_action_index.append([])
            self.terminate.append([])
        self.data_size = [0 for _ in range(argus.env.agent_num)]
        self.joint_data_size = 0
        self.clear_rate = argus.train.buffer_clear_rate

    def _store_exprience(self):
        if self.argus.train.joint_store:
            traj_len = np.min([len(self.traj_buffer.action_trajectory[agent_index]) for agent_index in range(self.argus.env.agent_num)])
            for index in range(traj_len - self.argus.train.TD):
                self._store(
                    s=np.vstack([self.traj_buffer.state_trajectory[agent_index][index] for agent_index in range(self.argus.env.agent_num)]),
                    a=np.vstack([self.traj_buffer.action_trajectory[agent_index][index] for agent_index in range(self.argus.env.agent_num)]),
                    a_p=np.vstack([self.traj_buffer.action_prob_trajectory[agent_index][index] for agent_index in range(self.argus.env.agent_num)]),
                    a_d=np.vstack([self.traj_buffer.action_dist_trajectory[agent_index][index] for agent_index in range(self.argus.env.agent_num)]),
                    r=np.vstack([calculate_TD_discount_reward(
                        self.traj_buffer.reward_trajectory[agent_index][index:index + self.argus.train.TD],
                        gamma=self.argus.train.gamma) for agent_index in range(self.argus.env.agent_num)]),
                    s_=np.vstack([self.traj_buffer.state_trajectory[agent_index][index + self.argus.train.TD] for agent_index in range(self.argus.env.agent_num)]),
                    a_=np.vstack([self.traj_buffer.action_trajectory[agent_index][index + self.argus.train.TD] for agent_index in range(self.argus.env.agent_num)]),
                    terminate=np.vstack([self.traj_buffer.terminate_trajectory[agent_index][index + self.argus.train.TD] for agent_index in range(self.argus.env.agent_num)]),
                )
        else:
            for agent_index in range(self.argus.env.agent_num):
                for index in range(len(self.traj_buffer.action_trajectory[agent_index]) - self.argus.train.TD):
                    self._store(
                        s=self.traj_buffer.state_trajectory[agent_index][index],
                        a=self.traj_buffer.action_trajectory[agent_index][index],
                        a_p=self.traj_buffer.action_prob_trajectory[agent_index][index],
                        a_d=self.traj_buffer.action_dist_trajectory[agent_index][index],
                        r=calculate_TD_discount_reward(self.traj_buffer.reward_trajectory[agent_index][index:index + self.argus.train.TD], gamma=self.argus.train.gamma),
                        s_=self.traj_buffer.state_trajectory[agent_index][index + self.argus.train.TD],
                        a_=self.traj_buffer.action_trajectory[agent_index][index + self.argus.train.TD],
                        terminate=self.traj_buffer.terminate_trajectory[agent_index][index + self.argus.train.TD],
                        agent_index=agent_index)
        self.traj_buffer.clear_traj_buffer()

    def _store(self, s, a, a_p, a_d, r, s_, a_, terminate, agent_index=None, **kwargs):
        if self.argus.train.joint_store:
            self.buffer_js.append(copy.deepcopy(np.reshape(s, [1, self.argus.env.agent_num, self.argus.env.state_dim])))
            self.joint_action_index.append(copy.deepcopy(np.reshape(a, [1, self.argus.env.agent_num, 1])))
            self.joint_action_prob.append(copy.deepcopy(np.reshape(a_p, [1, self.argus.env.agent_num, 1])))
            self.joint_action_dist.append(copy.deepcopy(np.reshape(a_d, [1, self.argus.env.agent_num, 1])))
            self.joint_reward.append(copy.deepcopy(np.reshape(r, [1, self.argus.env.agent_num, 1])))
            self.next_buffer_js.append(copy.deepcopy(np.reshape(s_, [1, self.argus.env.agent_num, self.argus.env.state_dim])))
            self.next_joint_action_index.append(copy.deepcopy(np.reshape(a_, [1, self.argus.env.agent_num, 1])))
            self.joint_terminate.append(copy.deepcopy(np.reshape(terminate, [1, self.argus.env.agent_num, 1])))
            self.joint_data_size += 1
        else:
            self.buffer_s[agent_index].append(np.reshape(s, [1, self.argus.env.state_dim]))
            self.action_index[agent_index].append(np.reshape(a, [1, 1]))
            self.action_prob[agent_index].append(np.reshape(a_p, [1, 1]))
            self.action_dist[agent_index].append(np.reshape(a_d, [1, 1]))
            self.reward[agent_index].append(np.reshape(r, [1, 1]))
            self.next_buffer_s[agent_index].append(np.reshape(s_, [1, self.argus.env.state_dim]))
            self.next_action_index[agent_index].append(np.reshape(a_, [1, 1]))
            self.terminate[agent_index].append(np.reshape(terminate, [1, 1]))
            self.data_size[agent_index] += 1

    def _get_experience(self, sample_num, select_whole_exp=False):
        if select_whole_exp:
            return (self.buffer_js, self.joint_action_index, self.joint_action_prob, self.joint_action_dist, self.joint_reward, self.next_buffer_js, self.next_joint_action_index, self.joint_terminate), self.joint_data_size
        else:
            if self.argus.train.joint_store:
                sample_num = self.joint_data_size if sample_num > self.joint_data_size else sample_num
                num_range = self.joint_data_size
            else:
                sample_num = np.min(self.data_size) if sample_num > np.min(self.data_size) else sample_num
                num_range = np.min(self.data_size)
            sample_index = self._create_sample_index(sample_num=sample_num, num_range=num_range)
            buffer_s, buffer_a, buffer_a_p, buffer_a_d, buffer_r, buffer_s_, buffer_a_, buffer_terminate = self._get_experience_with_index(sample_index=sample_index)
            return (buffer_s, buffer_a, buffer_a_p, buffer_a_d, buffer_r, buffer_s_, buffer_a_, buffer_terminate), sample_num

    def _create_sample_index(self, sample_num, num_range):
        if self.argus.train.joint_store:
            sample_index = [random.randint(0, num_range - 1) for _ in range(sample_num)]
        else:
            sample_index = [random.randint(0, num_range - 1) for _ in range(sample_num)]
        return sample_index

    def _get_experience_with_index(self, sample_index):
        buffer_s, buffer_a, buffer_a_p, buffer_a_d, buffer_r, buffer_s_, buffer_a_, buffer_terminate = [], [], [], [], [], [], [], []
        if self.argus.train.joint_store:
            for index in sample_index:
                buffer_s.append(self.buffer_js[index])
                buffer_a.append(self.joint_action_index[index])
                buffer_a_p.append(self.joint_action_prob[index])
                buffer_a_d.append(self.joint_action_dist[index])
                buffer_r.append(self.joint_reward[index])
                buffer_s_.append(self.next_buffer_js[index])
                buffer_a_.append(self.next_joint_action_index[index])
                buffer_terminate.append(self.joint_terminate[index])
        else:
            for agent_index in range(self.argus.env.agent_num):
                buffer_s.append([])
                buffer_a.append([])
                buffer_a_p.append([])
                buffer_a_d.append([])
                buffer_r.append([])
                buffer_s_.append([])
                buffer_a_.append([])
                buffer_terminate.append([])
                for index in sample_index:
                    buffer_s[agent_index].append(self.buffer_s[agent_index][index])
                    buffer_a[agent_index].append(self.action_index[agent_index][index])
                    buffer_a_p[agent_index].append(self.action_prob[agent_index][index])
                    buffer_a_d[agent_index].append(self.action_dist[agent_index][index])
                    buffer_r[agent_index].append(self.reward[agent_index][index])
                    buffer_s_[agent_index].append(self.next_buffer_s[agent_index][index])
                    buffer_a_[agent_index].append(self.next_action_index[agent_index][index])
                    buffer_terminate[agent_index].append(self.terminate[agent_index][index])
        return buffer_s, buffer_a, buffer_a_p, buffer_a_d, buffer_r, buffer_s_, buffer_a_, buffer_terminate

    def _buffer_reset(self, clear_all=False, clear_all_agent_buffer=False, **kwargs):
        if self.argus.train.joint_store:
            self._joint_buffer_reset(clear_all=clear_all)
        else:
            self._single_buffer_reset(clear_all=clear_all, clear_all_agent_buffer=clear_all_agent_buffer, **kwargs)

    def _single_buffer_reset(self, clear_all=True, clear_all_agent_buffer=False, **kwargs):
        if clear_all:
            if clear_all_agent_buffer:
                for agent_index in range(self.argus.env.agent_num):
                    self.buffer_s[agent_index].clear()
                    self.action_index[agent_index].clear()
                    self.action_prob[agent_index].clear()
                    self.action_dist[agent_index].clear()
                    self.reward[agent_index].clear()
                    self.next_buffer_s[agent_index].clear()
                    self.next_action_index[agent_index].clear()
                    self.terminate[agent_index].clear()
                    self._buffer_count_reset(agent_index=agent_index)
            else:
                self.buffer_s[kwargs["agent_index"]].clear()
                self.action_index[kwargs["agent_index"]].clear()
                self.action_prob[kwargs["agent_index"]].clear()
                self.action_dist[kwargs["agent_index"]].clear()
                self.reward[kwargs["agent_index"]].clear()
                self.next_buffer_s[kwargs["agent_index"]].clear()
                self.next_action_index[kwargs["agent_index"]].clear()
                self.terminate[kwargs["agent_index"]].clear()
                self._buffer_count_reset(agent_index=kwargs["agent_index"])
        else:
            if clear_all_agent_buffer:
                for agent_index in range(self.argus.env.agent_num):
                    clear_buffer_num = int(self.data_size[agent_index] * self.clear_rate)
                    del self.buffer_s[agent_index][0:clear_buffer_num]
                    del self.action_index[agent_index][0:clear_buffer_num]
                    del self.action_prob[agent_index][0:clear_buffer_num]
                    del self.action_dist[agent_index][0:clear_buffer_num]
                    del self.reward[agent_index][0:clear_buffer_num]
                    del self.next_buffer_s[agent_index][0:clear_buffer_num]
                    del self.next_action_index[agent_index][0:clear_buffer_num]
                    del self.terminate[agent_index][0:clear_buffer_num]
                    self._buffer_count_reset(agent_index=agent_index, clear_num=clear_buffer_num)
            else:
                clear_buffer_num = int(self.data_size[kwargs["agent_index"]] * self.clear_rate)
                del self.buffer_s[kwargs["agent_index"]][0:clear_buffer_num]
                del self.action_index[kwargs["agent_index"]][0:clear_buffer_num]
                del self.action_prob[kwargs["agent_index"]][0:clear_buffer_num]
                del self.action_dist[kwargs["agent_index"]][0:clear_buffer_num]
                del self.reward[kwargs["agent_index"]][0:clear_buffer_num]
                del self.next_buffer_s[kwargs["agent_index"]][0:clear_buffer_num]
                del self.next_action_index[kwargs["agent_index"]][0:clear_buffer_num]
                del self.terminate[kwargs["agent_index"]][0:clear_buffer_num]
                self._buffer_count_reset(agent_index=kwargs["agent_index"], clear_num=clear_buffer_num)

    def _joint_buffer_reset(self, clear_all=True):
        if clear_all:
            self.buffer_js.clear()
            self.joint_action_index.clear()
            self.joint_action_prob.clear()
            self.joint_action_dist.clear()
            self.joint_reward.clear()
            self.next_buffer_js.clear()
            self.next_joint_action_index.clear()
            self.joint_terminate.clear()
            self._joint_buffer_count_reset()
        else:
            clear_buffer_num = int(self.joint_data_size * self.clear_rate)
            del self.buffer_js[0:clear_buffer_num]
            del self.joint_action_index[0:clear_buffer_num]
            del self.joint_action_prob[0:clear_buffer_num]
            del self.joint_action_dist[0:clear_buffer_num]
            del self.joint_reward[0:clear_buffer_num]
            del self.next_buffer_js[0:clear_buffer_num]
            del self.next_joint_action_index[0:clear_buffer_num]
            del self.joint_terminate[0:clear_buffer_num]
            self._joint_buffer_count_reset(clear_num=clear_buffer_num)

    def _buffer_count_reset(self, agent_index, clear_num=None):
        if clear_num is not None:
            self.data_size[agent_index] -= clear_num
        else:
            self.data_size[agent_index] = 0

    def _joint_buffer_count_reset(self, clear_num=None):
        if clear_num is not None:
            self.joint_data_size -= clear_num
        else:
            self.joint_data_size = 0

    def get_buffer_size(self):
        if self.argus.train.joint_store:
            return self.joint_data_size
        else:
            return np.min(self.data_size)

class Replay_Buffer(bese_replay_buffer):
    def __init__(self, argus):
        super(Replay_Buffer, self).__init__(argus)

    def _store_exprience(self):
        traj_len = len(self.traj_buffer.state_trajectory)
        for index in range(traj_len - self.argus.train.TD):
            self._store(
                s=self.traj_buffer.state_trajectory[index],
                a=self.traj_buffer.action_trajectory[index],
                a_p=self.traj_buffer.action_prob_trajectory[index],
                a_d=self.traj_buffer.action_dist_trajectory[index],
                r=calculate_TD_discount_reward(
                    self.traj_buffer.reward_trajectory[index:index + self.argus.train.TD], gamma=self.argus.train.gamma),
                s_=self.traj_buffer.state_trajectory[index + self.argus.train.TD],
                a_=self.traj_buffer.action_trajectory[np.minimum(index + self.argus.train.TD, traj_len-2)],
                terminate=self.traj_buffer.terminate_trajectory[index + self.argus.train.TD],
            )
        self.traj_buffer._clear_traj_buffer()

    def _store(self, s, a, a_p, a_d, r, s_, a_, terminate,**kwargs):
        self.buffer_js.append(copy.deepcopy(np.reshape(s, [1, self.argus.env.agent_num, self.argus.env.state_dim])))
        self.joint_action_index.append(copy.deepcopy(np.reshape(a, [1, self.argus.env.agent_num, self.argus.env.n_action])))
        self.joint_action_prob.append(copy.deepcopy(np.reshape(a_p, [1, self.argus.env.agent_num, self.argus.env.n_action])))
        self.joint_action_dist.append(copy.deepcopy(np.reshape(a_d, [1, self.argus.env.agent_num, self.argus.env.action_dim])))
        self.joint_reward.append(copy.deepcopy(np.reshape(r, [1, self.argus.env.agent_num, 1])))
        self.next_buffer_js.append(copy.deepcopy(np.reshape(s_, [1, self.argus.env.agent_num, self.argus.env.state_dim])))
        self.next_joint_action_index.append(copy.deepcopy(np.reshape(a_, [1, self.argus.env.agent_num, self.argus.env.n_action])))
        self.joint_terminate.append(copy.deepcopy(np.reshape(terminate, [1, self.argus.env.agent_num, 1])))
        self.joint_data_size += 1