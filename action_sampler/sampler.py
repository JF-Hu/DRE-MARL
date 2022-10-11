
import numpy as np
import random


def sample_with_prob(softmax_action, argus):
    softmax_action = np.reshape(softmax_action, [-1, argus.env.action_dim])
    action_index, action_prob, action_dist = [], [], []
    apply_action = [[0 for _ in range(argus.env.action_dim)] for __ in  range(argus.env.agent_num)]
    action_index_list = [_ for _ in range(argus.env.action_dim)]
    for agent_index in range(len(softmax_action)):
        selected_action = random.choices(action_index_list, softmax_action[agent_index])[0]
        action_index.append(selected_action)
        apply_action[agent_index][selected_action] = 1
        action_prob.append(softmax_action[agent_index][selected_action])
        action_dist.append(softmax_action[agent_index])
    return action_index, apply_action, action_prob, action_dist

def epsilon_greedy(softmax_action, argus):
    if argus.env.action_type == "discrete":
        softmax_action = np.reshape(softmax_action, [-1, argus.env.action_dim])
        action_index, action_prob, action_dist = [], [], []
        apply_action = [[0 for _ in range(argus.env.action_dim)] for __ in  range(argus.env.agent_num)]
        for agent_index in range(len(softmax_action)):
            if (random.random() <= argus.train.greedy):
                selected_action = int(np.argmax(softmax_action[agent_index], axis=-1))
                action_index.append(selected_action)
                apply_action[agent_index][selected_action] = 1
                action_prob.append(softmax_action[agent_index][selected_action])
                action_dist.append(softmax_action[agent_index])
            else:
                selected_action = random.randint(0, argus.env.action_dim - 1)
                action_index.append(selected_action)
                apply_action[agent_index][selected_action] = 1
                action_prob.append(softmax_action[agent_index][selected_action])
                action_dist.append(softmax_action[agent_index])
    elif argus.env.action_type == "multi_discrete":
        softmax_action = np.reshape(softmax_action, [-1, argus.env.action_dim])
        split_actions = np.split(softmax_action, [argus.env.action_ncat[0]], axis=-1)
        action_index, action_prob, action_dist = [], [], []
        apply_action = [np.array([0 for _ in range(argus.env.action_dim)]) for __ in range(argus.env.agent_num)]
        for agent_index in range(len(softmax_action)):
            temp_action_index = []
            temp_action_prob = []
            if (random.random() <= argus.train.greedy):
                for ii in range(len(split_actions)):
                    selected_action = int(np.argmax(split_actions[ii][agent_index], axis=-1))
                    temp_action_index.append(selected_action)
                    apply_action[agent_index][selected_action + argus.env.action_ncat[ii] - argus.env.action_ncat[0]] = 1
                    temp_action_prob.append(
                        softmax_action[agent_index][selected_action + argus.env.action_ncat[ii] - argus.env.action_ncat[0]])
                    # temp_action_dist.append(split_actions[ii][agent_index])
            else:
                for ii in range(len(split_actions)):
                    selected_action = random.randint(0, argus.env.action_ncat[ii] - 1)
                    temp_action_index.append(selected_action)
                    apply_action[agent_index][selected_action + argus.env.action_ncat[ii] - argus.env.action_ncat[0]] = 1
                    temp_action_prob.append(
                        softmax_action[agent_index][selected_action + argus.env.action_ncat[ii] - argus.env.action_ncat[0]])
            action_index.append(np.reshape(np.array(temp_action_index), [1, -1]))
            action_prob.append(np.reshape(np.array(temp_action_prob), [1, -1]))
            action_dist.append(softmax_action[agent_index])
    else: raise Exception("action_type is wrong!")
    return action_index, apply_action, action_prob, action_dist

def deterministic_action(softmax_action, argus):
    if argus.env.action_type == "discrete":
        softmax_action = np.reshape(softmax_action, [-1, argus.env.action_dim])
        action_index, action_prob, action_dist = [], [], []
        apply_action = [[0 for _ in range(argus.env.action_dim)] for __ in range(argus.env.agent_num)]
        for agent_index in range(len(softmax_action)):
            selected_action = int(np.argmax(softmax_action[agent_index], axis=-1))
            action_index.append(selected_action)
            apply_action[agent_index][selected_action] = 1
            action_prob.append(softmax_action[agent_index][selected_action])
            action_dist.append(softmax_action[agent_index])
    elif argus.env.action_type == "multi_discrete":
        softmax_action = np.reshape(softmax_action, [-1, argus.env.action_dim])
        split_actions = np.split(softmax_action, [argus.env.action_ncat[0]], axis=-1)
        action_index, action_prob, action_dist = [], [], []
        apply_action = [np.array([0 for _ in range(argus.env.action_dim)]) for __ in range(argus.env.agent_num)]
        for agent_index in range(len(softmax_action)):
            temp_action_index = []
            temp_action_prob = []
            for ii in range(len(split_actions)):
                selected_action = int(np.argmax(split_actions[ii][agent_index], axis=-1))
                temp_action_index.append(selected_action)
                apply_action[agent_index][selected_action + argus.env.action_ncat[ii] - argus.env.action_ncat[0]] = 1
                temp_action_prob.append(
                    softmax_action[agent_index][selected_action + argus.env.action_ncat[ii] - argus.env.action_ncat[0]])
            action_index.append(np.reshape(np.array(temp_action_index), [1, -1]))
            action_prob.append(np.reshape(np.array(temp_action_prob), [1, -1]))
            action_dist.append(softmax_action[agent_index])
    else: raise Exception("action_type is wrong!")
    return action_index, apply_action, action_prob, action_dist


def simple_sampler(softmax_action, argus, evaluate, mode="epsilon_greedy"):
    if evaluate:
        return deterministic_action(softmax_action, argus)
    else:
        if mode == "prob":
            return sample_with_prob(softmax_action, argus)
        elif mode == "epsilon_greedy":
            return epsilon_greedy(softmax_action, argus)












