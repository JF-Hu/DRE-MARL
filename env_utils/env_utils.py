

import random
import numpy as np


def load_scenario(env_name="simple_spread", **kwargs):
    if kwargs["env_category"] == "MPE":
        if env_name == "simple_spread":
            from multiagent_particle_envs.multiagent.environment import MultiAgentEnv
            import multiagent_particle_envs.multiagent.scenarios as scenarios
            scenario = scenarios.load(env_name + ".py").Scenario()
            world = scenario.make_world()
            return MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                                 collide_reward=scenario.collide_reward)
        elif env_name == "simple_reference":
            from multiagent_particle_envs.multiagent.environment import MultiAgentEnv
            import multiagent_particle_envs.multiagent.scenarios as scenarios
            scenario = scenarios.load(env_name + ".py").Scenario()
            world = scenario.make_world()
            return MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
        elif env_name == "fullobs_collect_treasure":
            from multiagent_particle_envs.multiagent.environment_trea import MultiAgentEnv
            import multiagent_particle_envs.multiagent.scenarios as scenarios
            scenario = scenarios.load(env_name + ".py").Scenario()
            world = scenario.make_world()
            return MultiAgentEnv(
                world, reset_callback=scenario.reset_world, reward_callback=scenario.reward,
                observation_callback=scenario.observation, post_step_callback=scenario.post_step, discrete_action=True)
    else:
        raise Exception("Env domain error.")

def reward_uncertainty_process(reward_n, action_index, argus, env):
    if argus.reward.reward_uncertainty_type == "r_dete":
        if argus.reward.team_reward:
            pass
        else:
            for index, agent in enumerate(env.agents):
                reward_n[index] = env._get_reward(agent)
    elif argus.reward.reward_uncertainty_type == "r_dist":
        if argus.reward.team_reward:
            reward_random = np.random.normal(reward_n[0], 1) * 0.05
            for index in range(len(reward_n)):
                reward_n[index] += reward_random
        else:
            for index, agent in enumerate(env.agents):
                reward_n[index] = env._get_reward(agent) + np.random.normal(reward_n[index], 1) * 0.05
    elif argus.reward.reward_uncertainty_type == "r_ac-dist":
        reward_n.clear()
        for index, agent in enumerate(env.agents):
            if argus.env.scenario == "simple_reference":
                reward_n.append(env._get_reward(agent) + np.random.normal(action_index[index][0, 0], argus.reward.sdrr))
            else:
                reward_n.append(env._get_reward(agent) + np.random.normal(action_index[index], argus.reward.sdrr))
        if argus.reward.team_reward:
            reward_n = [np.sum(reward_n)] * argus.env.agent_num
    return reward_n

def reward_aggregation_process(reward_aggregation_type, dre_mix_reward, environmental_reward, argus):
    if reward_aggregation_type == "l_mo+g_mo":
        lumped_reward = np.tile(np.mean(np.mean(dre_mix_reward, axis=-1, keepdims=True), axis=-2, keepdims=True), (1, argus.env.agent_num, 1))
        mixed_reward = np.tile(np.mean(dre_mix_reward, axis=-2, keepdims=True), (1, argus.env.agent_num, 1))
    elif reward_aggregation_type == "l_smo+g_mo":
        lumped_reward = np.mean(dre_mix_reward, axis=-1, keepdims=True)
        mixed_reward = np.tile(np.mean(dre_mix_reward, axis=-2, keepdims=True), (1, argus.env.agent_num, 1))
    elif reward_aggregation_type == "l_ss+g_ss":
        lumped_reward = environmental_reward
        mixed_reward = dre_mix_reward
    elif reward_aggregation_type == "l_smo+g_ss":
        lumped_reward = np.mean(dre_mix_reward, axis=-1, keepdims=True)
        mixed_reward = dre_mix_reward
    elif reward_aggregation_type == "l_smo":
        lumped_reward = np.mean(dre_mix_reward, axis=-1, keepdims=True)
        mixed_reward = np.tile(np.mean(dre_mix_reward, axis=-1, keepdims=True), (1, 1, argus.env.action_dim))
    else:
        raise Exception("reward aggregation type error!")
    return lumped_reward, mixed_reward

def modify_learning_rate(ep, argus):
    if argus.train.lr_decay_interval is not None:
        if (ep+1) % argus.train.lr_decay_interval == 0:
            argus.critic.lr = np.maximum(argus.critic.lr * argus.train.decay_rate, argus.train.lr_min)
            argus.actor.lr = np.maximum(argus.actor.lr * argus.train.decay_rate, argus.train.lr_min)
            argus.reward.lr = np.maximum(argus.reward.lr * argus.train.decay_rate, argus.train.lr_min)
        else:
            argus.critic.lr = argus.critic.lr
            argus.actor.lr = argus.actor.lr
            argus.reward.lr = argus.reward.lr
    else:
        pass
    if argus.train.greedy_grow_interval is not None:
        if (ep+1) % argus.train.greedy_grow_interval == 0:
            argus.train.greedy = np.minimum(argus.train.greedy * argus.train.grow_rate, argus.train.greedy_max)
        else:
            argus.train.greedy = argus.train.greedy
    else:
        pass

def normalize_reward(reward, rate=1.0):
    if type(reward) == type([]):
        for index in range(len(reward)):
            reward[index] /= rate
        return reward
    else:
        return reward / rate

def get_total_reward(agents_reward, agent_num, reward_scale, curiosity_reward=None, **kwargs):
    if kwargs["domain"] == "MPE":
        try:
            agents_reward = np.reshape(agents_reward, [1, agent_num, 1])
        except:
            raise Exception("the shape of agents_reward is wrong!!!")
        if curiosity_reward is not None:
            assert agents_reward.shape == curiosity_reward.shape
            return agents_reward + curiosity_reward
        else:
            return agents_reward
    elif kwargs["domain"] == "SMAC":
        try:
            agents_reward = normalize_reward(
                np.reshape(np.array([agents_reward for _ in range(agent_num)]), [1, agent_num, 1]), rate=reward_scale)
        except:
            raise Exception("the shape of agents_reward is wrong!!!")
        if curiosity_reward is not None:
            assert agents_reward.shape == curiosity_reward.shape
            return agents_reward + curiosity_reward
        else:
            return agents_reward
    else:
        raise Exception("domain is wrong!!!")

def batch_iter(experiences, data_size, argus):
    total_sample_s, total_sample_a, total_sample_a_p, total_sample_a_d, total_sample_r, total_sample_s_, total_sample_a_, total_sample_terminate = experiences
    buffer_s_matrix, buffer_a_matrix, buffer_a_p_matrix, buffer_a_d_matrix, buffer_r_matrix, buffer_s__matrix, buffer_a__matrix, buffer_terminate_matrix = \
        np.vstack(total_sample_s),\
        np.vstack(total_sample_a),\
        np.vstack(total_sample_a_p),\
        np.vstack(total_sample_a_d),\
        np.vstack(total_sample_r),\
        np.vstack(total_sample_s_),\
        np.vstack(total_sample_a_),\
        np.vstack(total_sample_terminate)
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_buffer_s_matrix = buffer_s_matrix[shuffle_indices]
    shuffled_buffer_a_matrix = buffer_a_matrix[shuffle_indices]
    shuffled_buffer_a_p_matrix = buffer_a_p_matrix[shuffle_indices]
    shuffled_buffer_a_d_matrix = buffer_a_d_matrix[shuffle_indices]
    shuffled_buffer_r_matrix = buffer_r_matrix[shuffle_indices]
    shuffled_buffer_s__matrix = buffer_s__matrix[shuffle_indices]
    shuffled_buffer_a__matrix = buffer_a__matrix[shuffle_indices]
    shuffled_buffer_terminate_matrix= buffer_terminate_matrix[shuffle_indices]
    for batch_num in range(int((data_size - 1) / argus.train.mini_batch_size) + 1):
        start_index = batch_num * argus.train.mini_batch_size
        end_index = min((batch_num + 1) * argus.train.mini_batch_size, data_size)
        yield shuffled_buffer_s_matrix[start_index:end_index], \
              shuffled_buffer_a_matrix[start_index:end_index], \
              shuffled_buffer_a_p_matrix[start_index:end_index], \
              shuffled_buffer_a_d_matrix[start_index:end_index], \
              shuffled_buffer_r_matrix[start_index:end_index], \
              shuffled_buffer_s__matrix[start_index:end_index],\
              shuffled_buffer_a__matrix[start_index:end_index],\
              shuffled_buffer_terminate_matrix[start_index:end_index]



