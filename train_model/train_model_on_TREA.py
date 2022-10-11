import os
import time
import copy
import random
import numpy as np
import datetime
from algorithm.hyperpara_setting import get_variant_config
from models.MADRE_model import MADRE
from buffer.replay_buffer import Replay_Buffer
from train_info_recoder.MPE_data_recoder import CoopNavi_recorder as data_recoder
from env_utils import env_utils
from env_utils.env_utils import load_scenario
from evaluate.model_eval import evaluate_on_treasure_xcxb as evaluate_on_treasure
import wandb

def train_model(agent, argus, replay_buffer, ep, time_step):
    if argus.train.buffer_select_with_batch_size:
        experiences, data_size = replay_buffer._get_experience(sample_num=argus.train.batch_size)
    else:
        experiences, data_size = replay_buffer._get_experience(sample_num=int(replay_buffer.get_buffer_size()*argus.train.buffer_select_rate))
    batch_data_iter = env_utils.batch_iter(experiences=experiences, data_size=data_size, argus=argus)
    ave_critic_loss, ave_actor_loss, ave_reward_loss, ave_relative_ratio_var = [], [], [], []
    for batch in range(int((data_size - 1) / argus.train.mini_batch_size) + 1):
        batch_experiences = batch_data_iter.__next__()
        critic_loss, actor_loss, reward_loss, average_relative_ratio_var = agent._train(batch_experiences, argus=argus, ep=ep, time_step=time_step, env_category="MPE")
        ave_critic_loss.append(np.mean(critic_loss))
        ave_actor_loss.append(np.mean(actor_loss))
        ave_reward_loss.append(np.mean(reward_loss))
        ave_relative_ratio_var.append(np.mean(average_relative_ratio_var))
    return ave_critic_loss, ave_actor_loss, ave_reward_loss, ave_relative_ratio_var

def prepare_for_train(model, replay_buffer, train_recoder, env, argus):
    for ep in range(argus.train.before_train_episode):
        observation_list = env.reset()
        for pach_i in range(int(env.n/2)):
            observation_list[-pach_i-1] = np.concatenate([observation_list[-pach_i-1], np.array([1 for _ in range(3)])], axis=0)
        train_recoder.single_reward_recoder_reset()
        for t in range(argus.train.per_episode_length):
            train_recoder.time_step += 1
            replay_buffer.traj_buffer._store_state_data(
                data=copy.deepcopy(np.reshape(np.array(observation_list), [1, argus.env.agent_num, argus.env.state_dim])))
            replay_buffer.traj_buffer._store_terminate_data(
                data=copy.deepcopy(np.reshape(np.array([1 for _ in range(argus.env.agent_num)]), [1, argus.env.agent_num, 1])))
            action_index, apply_action, action_prob, action_dist = \
                model._action(agents_state=replay_buffer.traj_buffer.state_trajectory[-1], argus=argus, is_evaluate=False)
            replay_buffer.traj_buffer._store_action_data(data=copy.deepcopy(np.reshape(action_index, [1, argus.env.agent_num, 1])))
            replay_buffer.traj_buffer._store_action_prob_data(data=copy.deepcopy(np.reshape(action_prob, [1, argus.env.agent_num, 1])))
            replay_buffer.traj_buffer._store_action_dist_data(data=copy.deepcopy(np.reshape(action_dist, [1, argus.env.agent_num, argus.env.action_dim])))
            observation_list, reward_n, done_n, _ = env.step(apply_action)
            for pach_i in range(int(env.n/2)):
                observation_list[-pach_i - 1] = np.concatenate([observation_list[-pach_i-1], np.array([1 for _ in range(3)])], axis=0)
            train_recoder.single_reward_recoder(rewards=reward_n, scale_rate=argus.train.reward_scale, time_step_on_this_ep=t)
            if argus.train.single_step_mode:
                replay_buffer.traj_buffer._store_reward_data(
                    data=env_utils.get_total_reward(agents_reward=train_recoder.single_step_reward, agent_num=argus.env.agent_num,
                                                    reward_scale=argus.train.reward_scale, curiosity_reward=None, domain=argus.env.domain))
            else:
                replay_buffer.traj_buffer._store_reward_data(
                    data=env_utils.get_total_reward(agents_reward=reward_n, agent_num=argus.env.agent_num,
                                                    reward_scale=argus.train.reward_scale, curiosity_reward=None, domain=argus.env.domain))
            done = False
            if done_n[0] is True and done_n[1] is True and done_n[2] is True:
                done = True
            if done or (t >= argus.train.per_episode_length - 1):
                replay_buffer.traj_buffer._store_state_data(
                    data=copy.deepcopy(np.reshape(np.array(observation_list), [1, argus.env.agent_num, argus.env.state_dim])))
                replay_buffer.traj_buffer._store_terminate_data(
                    data=copy.deepcopy(np.reshape(np.array([0 for _ in range(argus.env.agent_num)]), [1, argus.env.agent_num, 1])))
                replay_buffer._store_exprience()
                train_recoder.store_ep_data(x_axis=train_recoder.time_step)
                if (ep+1)%argus.evaluate.eval_every_n_ep == 0:
                    eval_episode_rewards, eval_agents_rewards = evaluate_on_treasure(agent=model, env=env, argus=argus, render=argus.evaluate.render)
                    eval_episode_rewards = eval_episode_rewards
                    train_recoder.store_eval_ep_data(data=eval_episode_rewards, x_axis=train_recoder.time_step)
                break

def before_train():
    pass
def after_train():
    pass

def wandb_variant_clarify(exp_name, config):
    return {"exp_name": exp_name, "var_config": config}

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    algorithm, argus = get_variant_config(scenario="fullobs_collect_treasure")
    env = load_scenario(env_name=argus.env.scenario, env_category=argus.env.domain)
    madre = MADRE(algorithm, argus)
    replay_buffer = Replay_Buffer(argus=argus)
    train_recoder = data_recoder(agent_num=argus.env.agent_num)
    prepare_for_train(model=madre, replay_buffer=replay_buffer, train_recoder=train_recoder, env=env, argus=argus)
    for ep in range(argus.train.max_episodes):
        env_utils.modify_learning_rate(ep, argus=argus)
        observation_list = env.reset()
        for pach_i in range(int(env.n/2)):
            observation_list[-pach_i-1] = np.concatenate([observation_list[-pach_i-1], np.array([1 for _ in range(3)])], axis=0)
        train_recoder.single_reward_recoder_reset()
        for t in range(argus.train.per_episode_length):
            train_recoder.time_step += 1
            replay_buffer.traj_buffer._store_state_data(
                data=copy.deepcopy(np.reshape(np.array(observation_list), [1, argus.env.agent_num, argus.env.state_dim])))
            replay_buffer.traj_buffer._store_terminate_data(
                data=copy.deepcopy(np.reshape(np.array([1 for _ in range(argus.env.agent_num)]), [1, argus.env.agent_num, 1])))
            action_index, apply_action, action_prob, action_dist = \
                madre._action(agents_state=replay_buffer.traj_buffer.state_trajectory[-1], argus=argus, is_evaluate=False)
            replay_buffer.traj_buffer._store_action_data(data=copy.deepcopy(np.reshape(action_index, [1, argus.env.agent_num, 1])))
            replay_buffer.traj_buffer._store_action_prob_data(data=copy.deepcopy(np.reshape(action_prob, [1, argus.env.agent_num, 1])))
            replay_buffer.traj_buffer._store_action_dist_data(data=copy.deepcopy(np.reshape(action_dist, [1, argus.env.agent_num, argus.env.action_dim])))
            observation_list, reward_n, done_n, _ = env.step(apply_action)
            reward_n = env_utils.reward_uncertainty_process(reward_n=reward_n, action_index=action_index, argus=argus, env=env)
            for pach_i in range(int(env.n/2)):
                observation_list[-pach_i - 1] = np.concatenate([observation_list[-pach_i-1], np.array([1 for _ in range(3)])], axis=0)
            train_recoder.single_reward_recoder(rewards=reward_n, scale_rate=argus.train.reward_scale, time_step_on_this_ep=t)
            if argus.train.single_step_mode:
                replay_buffer.traj_buffer._store_reward_data(
                    data=env_utils.get_total_reward(agents_reward=train_recoder.single_step_reward,
                                                    agent_num=argus.env.agent_num,
                                                    reward_scale=argus.train.reward_scale, curiosity_reward=None,
                                                    domain=argus.env.domain))
            else:
                replay_buffer.traj_buffer._store_reward_data(
                    data=env_utils.get_total_reward(agents_reward=reward_n, agent_num=argus.env.agent_num,
                                                    reward_scale=argus.train.reward_scale, curiosity_reward=None,
                                                    domain=argus.env.domain))
            done = False
            if done_n[0] is True and done_n[1] is True and done_n[2] is True:
                done = True
            if train_recoder.time_step % argus.save.save_data_every_n_step == 0:
                train_recoder.save_to_file(path=madre.exp_save_path)
            if done or (t >= argus.train.per_episode_length - 1):
                replay_buffer.traj_buffer._store_state_data(
                    data=copy.deepcopy(np.reshape(np.array(observation_list), [1, argus.env.agent_num, argus.env.state_dim])))
                replay_buffer.traj_buffer._store_terminate_data(data=copy.deepcopy(np.reshape(np.array([0 for _ in range(argus.env.agent_num)]), [1, argus.env.agent_num, 1])))
                train_recoder.store_ep_data(x_axis=train_recoder.time_step)
                replay_buffer._store_exprience()
                if (ep+1)%argus.evaluate.eval_every_n_ep == 0:
                    eval_episode_rewards, eval_agents_rewards = evaluate_on_treasure(agent=madre, env=env, argus=argus, render=argus.evaluate.render)
                    if eval_episode_rewards > argus.control.render_threshold:
                        argus.evaluate.render = False
                    else:
                        argus.evaluate.render = False
                    print("{:*>20} eval_ep_rewards: {}, eval_ag_rewards: {}".format("EvalInfo:", eval_episode_rewards, eval_agents_rewards))
                if train_recoder.time_step % argus.control.print_info_every_n_step == 0:
                    print("{:*>10} ep: {}, time_step: {} buffer_size: {} save_path: {}".format("GlobalInfo:", ep, train_recoder.time_step, replay_buffer.get_buffer_size(), madre.exp_save_path))
                    print("{:*>10} lr: {}, greedy: {}".format("GlobalInfo:", argus.critic.lr, argus.train.greedy))
                if (ep + 1) % argus.control.update_every_n_ep == 0:
                    ave_critic_loss, ave_actor_loss, ave_reward_loss, ave_relative_ratio_var = [], [], [], []
                    for _ in range(argus.control.update_repeat_n_time):
                        critic_loss, actor_loss, reward_loss, relative_ratio_var = train_model(
                            agent=madre, argus=argus, replay_buffer=replay_buffer, ep=ep,
                            time_step=train_recoder.time_step)
                        ave_critic_loss.append(critic_loss)
                        ave_actor_loss.append(actor_loss)
                        ave_reward_loss.append(reward_loss)
                        ave_relative_ratio_var.append(relative_ratio_var)
                    print("{:*>20} critic_loss: {}, actor_loss: {}, reward_loss: {}".format("TrainInfo:", ave_critic_loss, ave_actor_loss, ave_reward_loss))
                    for agent_index in range(argus.env.agent_num):
                        madre._update_parameters(agent_index=agent_index, net_type="policy")
                    madre._update_parameters(net_type="critic")
                if (ep + 1) % argus.control.clear_buffer_every_n_ep == 0:
                    replay_buffer._buffer_reset()
                if (ep + 1) % argus.control.plot_every_n_step == 0:
                    train_recoder.plot_ep_result(save_fic_path=madre.exp_save_path+"/result_pic", add_name=(madre.exp_save_path).split("/")[-1])
                break
