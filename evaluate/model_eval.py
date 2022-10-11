import copy
import numpy as np
from env_utils import env_utils
from buffer.replay_buffer import Replay_Buffer
import time

def evaluate_on_CoopNavi(agent, env, argus, render):
    replay_buffer = Replay_Buffer(argus=argus)
    episode_rewards = []
    agent_rewards = [[] for _ in range(env.n)]
    time_step = 0
    for ep in range(argus.evaluate.eval_episode):
        observation_list = env.reset()
        for t in range(25):
            if render:
                env.render()
                time.sleep(0.01)
            time_step += 1
            # normal interact with environment
            replay_buffer.traj_buffer._store_state_data(
                data=copy.deepcopy(np.reshape(np.array(observation_list), [1, argus.env.agent_num, argus.env.state_dim])))
            action_index, apply_action, action_prob, action_dist = \
                agent._action(agents_state=replay_buffer.traj_buffer.state_trajectory[-1], argus=argus, is_evaluate=True)
            observation_list, reward_n, done_n, _ = env.step(apply_action)
            if argus.reward.eval_with_reward_uncertainty:
                reward_n = env_utils.reward_uncertainty_process(reward_n=reward_n, action_index=action_index, argus=argus, env=env)
            done = False
            if done_n[0] is True and done_n[1] is True and done_n[2] is True:
                done = True
            replay_buffer.traj_buffer._store_reward_data(
                data=env_utils.get_total_reward(
                    agents_reward=reward_n, agent_num=argus.env.agent_num,
                    reward_scale=argus.train.reward_scale, curiosity_reward=None, domain=argus.env.domain))
            if done or (t >= 25 - 1):
                episode_rewards.append(np.sum(replay_buffer.traj_buffer.reward_trajectory))
                episode_rewards_traj = np.vstack(replay_buffer.traj_buffer._get_reward_traj())
                for agent_index in range(argus.env.agent_num):
                    agent_rewards[agent_index].append(np.sum(episode_rewards_traj[:, agent_index, :]))
                replay_buffer.traj_buffer._clear_traj_buffer()
                break
    return np.mean(episode_rewards), [np.mean(agent_rewards[agent_index]) for agent_index in range(argus.env.agent_num)]

def evaluate_on_reference(agent, env, argus, render):
    replay_buffer = Replay_Buffer(argus=argus)
    episode_rewards = []
    agent_rewards = [[] for _ in range(env.n)]
    time_step = 0
    for ep in range(argus.evaluate.eval_episode):
        observation_list = env.reset()
        for t in range(25):
            if render:
                env.render()
                time.sleep(0.01)
            time_step += 1
            # normal interact with environment
            replay_buffer.traj_buffer._store_state_data(
                data=copy.deepcopy(np.reshape(np.array(observation_list), [1, argus.env.agent_num, argus.env.state_dim])))
            action_index, apply_action, action_prob, action_dist = \
                agent._action(agents_state=replay_buffer.traj_buffer.state_trajectory[-1], argus=argus, is_evaluate=True)
            observation_list, reward_n, done_n, _ = env.step(apply_action)
            if argus.reward.eval_with_reward_uncertainty:
                reward_n = env_utils.reward_uncertainty_process(reward_n=reward_n, action_index=action_index, argus=argus, env=env)
            done = False
            if done_n[0] is True and done_n[1] is True and done_n[2] is True:
                done = True
            replay_buffer.traj_buffer._store_reward_data(
                data=env_utils.get_total_reward(
                    agents_reward=reward_n, agent_num=argus.env.agent_num,
                    reward_scale=argus.train.reward_scale, curiosity_reward=None, domain=argus.env.domain))
            if done or (t >= 25 - 1):
                episode_rewards.append(np.sum(replay_buffer.traj_buffer.reward_trajectory))
                episode_rewards_traj = np.vstack(replay_buffer.traj_buffer._get_reward_traj())
                for agent_index in range(argus.env.agent_num):
                    agent_rewards[agent_index].append(np.sum(episode_rewards_traj[:, agent_index, :]))
                replay_buffer.traj_buffer._clear_traj_buffer()
                break
    return np.mean(episode_rewards), [np.mean(agent_rewards[agent_index]) for agent_index in range(argus.env.agent_num)]

def evaluate_on_treasure_xcxb(agent, env, argus, render):
    replay_buffer = Replay_Buffer(argus=argus)
    episode_rewards = []
    agent_rewards = [[] for _ in range(env.n)]
    time_step = 0
    for ep in range(argus.evaluate.eval_episode):
        observation_list = env.reset()
        for pach_i in range(int(env.n/2)):
            observation_list[-pach_i - 1] = np.concatenate([observation_list[-pach_i - 1], np.array([1 for _ in range(3)])], axis=0)
        for t in range(argus.train.per_episode_length):
            if render:
                env.render()
                time.sleep(0.01)
            time_step += 1
            # normal interact with environment
            replay_buffer.traj_buffer._store_state_data(
                data=copy.deepcopy(np.reshape(np.array(observation_list), [1, argus.env.agent_num, argus.env.state_dim])))
            action_index, apply_action, action_prob, action_dist = \
                agent._action(agents_state=replay_buffer.traj_buffer.state_trajectory[-1], argus=argus, is_evaluate=True)
            observation_list, reward_n, done_n, _ = env.step(apply_action)
            if argus.reward.eval_with_reward_uncertainty:
                reward_n = env_utils.reward_uncertainty_process(reward_n=reward_n, action_index=action_index, argus=argus, env=env)
            for pach_i in range(int(env.n/2)):
                observation_list[-pach_i - 1] = np.concatenate([observation_list[-pach_i - 1], np.array([1 for _ in range(3)])], axis=0)
            done = False
            if done_n[0] is True and done_n[1] is True and done_n[2] is True:
                done = True
            replay_buffer.traj_buffer._store_reward_data(
                data=env_utils.get_total_reward(
                    agents_reward=reward_n, agent_num=argus.env.agent_num,
                    reward_scale=argus.train.reward_scale, curiosity_reward=None, domain=argus.env.domain))
            if done or (t >= argus.train.per_episode_length - 1):
                episode_rewards.append(np.sum(replay_buffer.traj_buffer.reward_trajectory))
                episode_rewards_traj = np.vstack(replay_buffer.traj_buffer._get_reward_traj())
                for agent_index in range(argus.env.agent_num):
                    agent_rewards[agent_index].append(np.sum(episode_rewards_traj[:, agent_index, :]))
                replay_buffer.traj_buffer._clear_traj_buffer()
                break
    return np.mean(episode_rewards), [np.mean(agent_rewards[agent_index]) for agent_index in range(argus.env.agent_num)]

