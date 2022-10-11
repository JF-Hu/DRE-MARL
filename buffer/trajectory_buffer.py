import copy
import numpy as np

class base_trajectory_buffer():
    def __init__(self, agent_num):
        self.agent_num = agent_num
        self.state_trajectory = []
        self.action_trajectory = []
        self.reward_trajectory = []
        self.action_prob_trajectory = []
        self.action_dist_trajectory = []
        self.action_predict_dist_trajectory = []
        self.lstm_roll_output = []
        for index in range(agent_num):
            self.state_trajectory.append([])
            self.action_trajectory.append([])
            self.reward_trajectory.append([])
            self.action_prob_trajectory.append([])
            self.action_dist_trajectory.append([])
            self.action_predict_dist_trajectory.append([])
            self.lstm_roll_output.append([[], [], []])

    def _get_stacked_state(self):
        return np.expand_dims(
            np.vstack([single_s_traj[-1] for single_s_traj in self.state_trajectory]),
            axis=0)

    def clear_traj_buffer(self):
        for index in range(self.agent_num):
            self.state_trajectory[index].clear()
            self.action_trajectory[index].clear()
            self.reward_trajectory[index].clear()
            self.action_prob_trajectory[index].clear()
            self.action_dist_trajectory[index].clear()
            self.action_predict_dist_trajectory[index].clear()
    def clear_single_traj_buffer(self, agent_index):
        self.state_trajectory[agent_index].clear()
        self.action_trajectory[agent_index].clear()
        self.reward_trajectory[agent_index].clear()
        self.action_prob_trajectory[agent_index].clear()
        self.action_dist_trajectory[agent_index].clear()
        self.action_predict_dist_trajectory[agent_index].clear()
    def store_state_data(self, data, agent_index):
        self.state_trajectory[agent_index].append(copy.deepcopy(data))
    def store_action_data(self, data, agent_index):
        self.action_trajectory[agent_index].append(copy.deepcopy(data))
    def store_reward_data(self, data, agent_index):
        self.reward_trajectory[agent_index].append(copy.deepcopy(data))
    def store_action_prob_data(self, data, agent_index):
        self.action_prob_trajectory[agent_index].append(copy.deepcopy(data))
    def store_action_dist_data(self, data, agent_index):
        self.action_dist_trajectory[agent_index].append(copy.deepcopy(data))
    def store_action_predict_dist_data(self, data, agent_index):
        self.action_predict_dist_trajectory[agent_index].append(copy.deepcopy(data))
    def get_state_traj(self, agent_index):
        return self.state_trajectory[agent_index]
    def get_action_traj(self, agent_index):
        return self.action_trajectory[agent_index]
    def get_reward_traj(self, agent_index):
        return self.reward_trajectory[agent_index]
    def get_action_prob_traj(self, agent_index):
        return self.action_prob_trajectory[agent_index]
    def get_action_dist_traj(self, agent_index):
        return self.action_dist_trajectory[agent_index]
    def get_action_predict_dist_traj(self, agent_index):
        return self.action_predict_dist_trajectory[agent_index]
    def clear_lstm_roll_buffer(self, LSTMFlags):
        for index in range(len(self.lstm_roll_output)):
            self.lstm_roll_output[index][0].clear()
            self.lstm_roll_output[index][1].clear()
            self.lstm_roll_output[index][2].clear()
            self.lstm_roll_output[index][1].append(np.array(np.zeros((1, LSTMFlags.LSTM_hidden_neural_size))))
            self.lstm_roll_output[index][2].append(np.array(np.zeros((1, LSTMFlags.LSTM_hidden_neural_size))))
    def store_lstm_roll_buffer(self, agent_index, state, cell_state, hidden_state):
        if state is not None:
            self.lstm_roll_output[agent_index][0].clear()
            self.lstm_roll_output[agent_index][0].append(copy.deepcopy(state))
        if cell_state is not None:
            self.lstm_roll_output[agent_index][1].clear()
            self.lstm_roll_output[agent_index][1].append(copy.deepcopy(cell_state))
        if hidden_state is not None:
            self.lstm_roll_output[agent_index][2].clear()
            self.lstm_roll_output[agent_index][2].append(copy.deepcopy(hidden_state))
    def get_lstm_roll_buffer(self, agent_index):
        return self.lstm_roll_output[agent_index]

class base_joint_trajectory_uffer():
    def __init__(self, agent_num):
        self.agent_num = agent_num
        self.state_trajectory = []
        self.action_trajectory = []
        self.reward_trajectory = []
        self.terminate_trajectory = []
        self.collide_trajectory = []
        self.action_prob_trajectory = []
        self.action_dist_trajectory = []
        self.action_predict_dist_trajectory = []
        self.lstm_roll_output = []
        for agent_index in range(agent_num):
            self.lstm_roll_output.append([[], []])

    def _get_stacked_state(self):
        return np.expand_dims(
            np.vstack([single_s_traj[-1] for single_s_traj in self.state_trajectory]),
            axis=0)

    def _clear_traj_buffer(self):
        self.state_trajectory.clear()
        self.action_trajectory.clear()
        self.reward_trajectory.clear()
        self.terminate_trajectory.clear()
        self.collide_trajectory.clear()
        self.action_prob_trajectory.clear()
        self.action_dist_trajectory.clear()
        self.action_predict_dist_trajectory.clear()

    def _store_state_data(self, data):
        self.state_trajectory.append(copy.deepcopy(data))
    def _store_action_data(self, data):
        self.action_trajectory.append(copy.deepcopy(data))
    def _store_reward_data(self, data):
        self.reward_trajectory.append(copy.deepcopy(data))
    def _store_terminate_data(self, data):
        self.terminate_trajectory.append(copy.deepcopy(data))
    def _store_collide_data(self, data):
        self.collide_trajectory.append(copy.deepcopy(data))
    def _store_action_prob_data(self, data):
        self.action_prob_trajectory.append(copy.deepcopy(data))
    def _store_action_dist_data(self, data):
        self.action_dist_trajectory.append(copy.deepcopy(data))
    def _store_action_predict_dist_data(self, data):
        self.action_predict_dist_trajectory.append(copy.deepcopy(data))
    def _get_state_traj(self):
        return self.state_trajectory
    def _get_action_traj(self):
        return self.action_trajectory
    def _get_reward_traj(self):
        return self.reward_trajectory
    def _get_terminate_traj(self):
        return self.terminate_trajectory
    def _get_collide_traj(self):
        return self.collide_trajectory
    def _get_action_prob_traj(self):
        return self.action_prob_trajectory
    def _get_action_dist_traj(self):
        return self.action_dist_trajectory
    def _get_action_predict_dist_traj(self):
        return self.action_predict_dist_trajectory
    def _clear_lstm_roll_buffer(self, LSTMFlags):
        for agent_index in range(self.agent_num):
            self.lstm_roll_output[agent_index][0].clear()
            self.lstm_roll_output[agent_index][1].clear()
            self.lstm_roll_output[agent_index][0].append(np.array(np.zeros((1, LSTMFlags.gru_hidden_neural))))
            self.lstm_roll_output[agent_index][1].append(np.array(np.zeros((1, LSTMFlags.gru_hidden_neural))))
    def _store_lstm_roll_buffer(self, cell_state, hidden_state):
        if cell_state is not None:
            for agent_index in range(self.agent_num):
                self.lstm_roll_output[agent_index][0].clear()
                self.lstm_roll_output[agent_index][0].append(copy.deepcopy(cell_state[agent_index]))
        if hidden_state is not None:
            for agent_index in range(self.agent_num):
                self.lstm_roll_output[agent_index][1].clear()
                self.lstm_roll_output[agent_index][1].append(copy.deepcopy(hidden_state[agent_index]))
    def _get_lstm_roll_buffer(self, LSTMFlags, cell_state=False, hidden_state=False):
        result = []
        if cell_state:
            result.append(np.reshape(
                np.vstack([self.lstm_roll_output[agent_index][0][0] for agent_index in range(self.agent_num)]), [1, self.agent_num, LSTMFlags.gru_hidden_neural]))
        if hidden_state:
            result.append(np.reshape(
                np.vstack([self.lstm_roll_output[agent_index][1][0] for agent_index in range(self.agent_num)]), [1, self.agent_num, LSTMFlags.gru_hidden_neural]))
        if len(result) == 0:
            return None
        elif len(result) == 1:
            return result[0]
        else:
            return tuple(result)



class joint_trajectory_buffer(base_joint_trajectory_uffer):
    def __init__(self, agent_num):
        super(joint_trajectory_buffer, self).__init__(agent_num)
        self.collide_trajectory = []

    def clear_traj_buffer(self):
        super(joint_trajectory_buffer, self)._clear_traj_buffer()
        self.collide_trajectory.clear()

    def store_collide_data(self, data):
        self.collide_trajectory.append(copy.deepcopy(data))

    def get_collide_traj(self):
        return self.collide_trajectory

class trajectory_buffer(base_trajectory_buffer):
    def __init__(self, agent_num):
        super(trajectory_buffer, self).__init__(agent_num)
        self.collide_trajectory = []
        for index in range(agent_num):
            self.collide_trajectory.append([])

    def clear_traj_buffer(self):
        super(trajectory_buffer, self).clear_traj_buffer()
        for index in range(self.agent_num):
            self.collide_trajectory[index].clear()

    def store_collide_data(self, data, agent_index):
        self.collide_trajectory[agent_index].append(copy.deepcopy(data))

    def get_collide_traj(self, agent_index):
        return self.collide_trajectory[agent_index]
