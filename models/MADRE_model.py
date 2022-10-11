
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import datetime
import json
import os
from tensorflow.keras.utils import to_categorical
from .actor import decentralized_actor
from .critic import centralized_critic
from action_sampler.sampler import simple_sampler
from .base_model import base_model
from env_utils.env_utils import reward_aggregation_process

class MADRE(base_model):
    def __init__(self, model_name, argus):
        super(MADRE, self).__init__()
        self.model_name = model_name
        self.argus = argus
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(config=tf_config, graph=self.graph)
        with self.graph.as_default():
            self.centralized_critic = centralized_critic(model_name="centralized_critic", argus=self.argus)
            self.decentralized_actor = decentralized_actor(model_name="decentralized_actor", argus=self.argus)
            self.verbose_info = self.generate_verbose_variant()
            if "/" in os.getcwd():
                split_item = "/"
            else:
                split_item = "\\"
            if os.getcwd().split(split_item)[-1] == "train_model":
                self.exp_save_path = "{}/train_result/exp_{}".format(os.getcwd(),
                    "%02d%2d-%2d-%2d-%2d" % (datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))
            else:
                self.exp_save_path = "{}/train_result/exp_{}".format(
                    os.getcwd()+"/train_model", "%02d%2d-%2d-%2d-%2d" % (
                        datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))
            if argus.control.summary_record:
                self.contribute_summary(
                    sess=self.sess,
                    summary_path=self.exp_save_path,
                    summary_variable_list=[value["tf_var"] for value in self.verbose_info.values()],
                    variable_name_list=[key for key in self.verbose_info.keys()])
            self.recoder_config(config_data=self.argus.config_record, path=self.exp_save_path)
            self.sess.run(tf.global_variables_initializer())
            for agent_index in range(self.argus.env.agent_num):
                self._reload_parameters(agent_index=agent_index, net_type="policy")
            self._reload_parameters(net_type="critic")

    def verbose_variant_update(self, verbose_info, wait_for_add_variant):
        index = 0
        for variant in wait_for_add_variant:
            for key,value in variant.items():
                verbose_info[key] = value
                verbose_info[key]["summary_index"] = index
                index += 1
        return verbose_info

    def generate_verbose_variant(self):
        policy_verbose_info = {"policy{}_loss".format(i): {"tf_var":self.decentralized_actor.policies[i].network_loss, "summary_index": None} for i in range(self.argus.env.agent_num)}
        reward_verbose_info = {"reward{}_loss".format(i): {"tf_var": self.decentralized_actor.policies[i].reward_pre_loss, "summary_index": None} for i in range(self.argus.env.agent_num)}
        critic_verbose_info = {"critic_loss": {"tf_var":self.centralized_critic.critic_network_loss, "summary_index": None}}
        verbose_info = self.verbose_variant_update(verbose_info={}, wait_for_add_variant=[policy_verbose_info, reward_verbose_info, critic_verbose_info])
        return verbose_info

    def check_path(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def recoder_config(self, config_data, path):
        self.check_path(path)
        with open(path+"/config.json", "w", encoding='utf8') as f:
            json.dump(config_data, f)
            f.close()

    def contribute_summary(self, sess, summary_path, summary_variable_list, variable_name_list):
        self.check_path(summary_path)
        assert isinstance(summary_variable_list, list)
        assert isinstance(variable_name_list, list)
        for variable_index in range(len(summary_variable_list)):
            self.summary.append(
                tf.summary.merge(
                    [tf.summary.scalar(
                        variable_name_list[variable_index], summary_variable_list[variable_index])]))
        self.summary_writer = tf.summary.FileWriter(summary_path+"/log", sess.graph)

    def _update_parameters(self, agent_index=0, net_type="policy", update_all=False):
        if update_all:
            if net_type == "policy":
                for i in range(self.argus.env.agent_num):
                    self.decentralized_actor._update_parameters(sess=self.sess, agent_index=i)
            else:
                self.centralized_critic._update_parameters(sess=self.sess)
        else:
            if net_type == "policy":
                self.decentralized_actor._update_parameters(sess=self.sess, agent_index=agent_index)
            else:
                self.centralized_critic._update_parameters(sess=self.sess)

    def _reload_parameters(self, agent_index=None, net_type="policy", reload_all=False):
        if reload_all:
            if net_type == "policy":
                for i in range(self.argus.env.agent_num):
                    self.decentralized_actor._reload_parameters(sess=self.sess, agent_index=i)
            else:
                self.centralized_critic._reload_parameters(sess=self.sess)
        else:
            if net_type == "policy":
                self.decentralized_actor._reload_parameters(sess=self.sess, agent_index=agent_index)
            else:
                self.centralized_critic._reload_parameters(sess=self.sess)

    def _action(self, agents_state, argus, is_evaluate=False):
        softmax_action = []
        for agent_index in range(self.argus.env.agent_num):
            if is_evaluate:
                action_prob = self.sess.run(
                    self.decentralized_actor.policies[agent_index].current_actor.outputs,
                    feed_dict={self.decentralized_actor.policies[agent_index].current_actor.inputs:agents_state[:, agent_index, :]})
            else:
                action_prob = self.sess.run(
                    self.decentralized_actor.policies[agent_index].target_actor.outputs,
                    feed_dict={self.decentralized_actor.policies[agent_index].target_actor.inputs: agents_state[:, agent_index, :]})
            softmax_action.append(action_prob)
        return simple_sampler(softmax_action, argus, is_evaluate)

    def _contrib_run_object(self, *args):
        expect_run_onject_list = []
        for obj in args:
            expect_run_onject_list.append(obj)
        return expect_run_onject_list

    def train_reward_only(self, experiences, **kwargs):
        stacked_s, stacked_a, stacked_a_p, stacked_a_d, stacked_r, stacked_s_, stacked_a_, stacked_terminate = experiences
        self.pre_reward_train(
            s=stacked_s,
            a=stacked_a,
            a_d=stacked_a_d,
            r=stacked_r,
            s_=stacked_s_,
            terminate=stacked_terminate,
            decentralized_policy=self.decentralized_actor.policies,
            time_step=kwargs["time_step"])

    def pre_reward_train(self, s, a, r, decentralized_policy, **kwargs):
        for agent_index in range(self.argus.env.agent_num):
            mix_reward, pre_reward_loss, summary_pre_reward_loss, _ = self.sess.run(
                self._contrib_run_object(
                    decentralized_policy[agent_index].mix_reward,
                    decentralized_policy[agent_index].reward_pre_loss,
                    self.summary[self.verbose_info["reward{}_loss".format(agent_index)]["summary_index"]],
                    decentralized_policy[agent_index].reward_train_op),
                feed_dict={
                    decentralized_policy[agent_index].reward_pre.inputs: s[:, agent_index, :],
                    decentralized_policy[agent_index].ref_reward: r[:, agent_index, :],
                    decentralized_policy[agent_index].onehot_action: to_categorical(a[:, agent_index, :],
                                                                                    num_classes=self.argus.env.action_dim)
                })
            self.summary_writer.add_summary(summary_pre_reward_loss, global_step=kwargs["time_step"])

    def _train(self, experiences, **kwargs):
        agents_critic_network_loss, agents_actor_network_loss, agents_pre_reward_loss, agents_relative_ratio_var = [], [], [], []
        stacked_s, stacked_a, stacked_a_p, stacked_a_d, stacked_r, stacked_s_, stacked_a_, stacked_terminate = experiences
        if self.argus.env.action_type == "discrete":
            advantage, critic_network_loss, summary_critic_loss, agents_pre_reward_loss = self._critic_train(
                s=stacked_s,
                a=stacked_a,
                a_d=stacked_a_d,
                r=stacked_r,
                s_=stacked_s_,
                terminate=stacked_terminate,
                decentralized_policy=self.decentralized_actor.policies,
                time_step=kwargs["time_step"],
                argus=kwargs["argus"]
            )
            self.summary_writer.add_summary(summary_critic_loss, global_step=kwargs["time_step"])
            for agent_index in range(self.argus.env.agent_num):
                actor_network_loss, summary_actor_loss, relative_ratio_var = self._policy_train(
                    s=stacked_s[:, agent_index, :],
                    a=stacked_a[:, agent_index, :],
                    a_p=stacked_a_p[:, agent_index, :],
                    r=stacked_r[:, agent_index, :],
                    s_=stacked_s_[:, agent_index, :],
                    advantage=np.reshape(advantage[:, agent_index, :], [-1, 1]),
                    decentralized_policy=self.decentralized_actor.policies[agent_index],
                    summary_var=self.summary[self.verbose_info["policy{}_loss".format(agent_index)]["summary_index"]]
                )
                self.summary_writer.add_summary(summary_actor_loss, global_step=kwargs["time_step"])
                agents_critic_network_loss.append(critic_network_loss)
                agents_actor_network_loss.append(actor_network_loss)
                agents_relative_ratio_var.append(relative_ratio_var)
            average_agents_relative_ratio_var = np.mean(agents_relative_ratio_var)
            return agents_critic_network_loss, agents_actor_network_loss, agents_pre_reward_loss, average_agents_relative_ratio_var
        else:
            advantage, critic_network_loss, summary_critic_loss, agents_pre_reward_loss = self._critic_train_multi_discrete(
                s=stacked_s,
                a=stacked_a,
                a_d=stacked_a_d,
                r=stacked_r,
                s_=stacked_s_,
                terminate=stacked_terminate,
                decentralized_policy=self.decentralized_actor.policies,
                time_step=kwargs["time_step"],
                argus=kwargs["argus"]
            )
            self.summary_writer.add_summary(summary_critic_loss, global_step=kwargs["time_step"])
            for agent_index in range(self.argus.env.agent_num):
                actor_network_loss, summary_actor_loss, relative_ratio_var = self._policy_train_multi_discrete(
                    s=stacked_s[:, agent_index, :],
                    a=stacked_a[:, agent_index, :],
                    a_p=stacked_a_p[:, agent_index, :],
                    r=stacked_r[:, agent_index, :],
                    s_=stacked_s_[:, agent_index, :],
                    advantage=np.reshape(advantage[:, agent_index, :], [-1, 1]),
                    decentralized_policy=self.decentralized_actor.policies[agent_index],
                    summary_var=self.summary[self.verbose_info["policy{}_loss".format(agent_index)]["summary_index"]]
                )
                self.summary_writer.add_summary(summary_actor_loss, global_step=kwargs["time_step"])
                agents_critic_network_loss.append(critic_network_loss)
                agents_actor_network_loss.append(actor_network_loss)
                agents_relative_ratio_var.append(relative_ratio_var)
            average_agents_relative_ratio_var = np.mean(agents_relative_ratio_var)
            return agents_critic_network_loss, agents_actor_network_loss, agents_pre_reward_loss, average_agents_relative_ratio_var

    def _policy_train(self, s, a, a_p, r, s_, advantage, decentralized_policy, summary_var, **kwargs):
        if summary_var is not None:
            actor_network_loss, summary_actor_loss, _, relative_ratio_var = self.sess.run(
                self._contrib_run_object(
                    decentralized_policy.network_loss,
                    summary_var,
                    decentralized_policy.actor_train_op,
                    decentralized_policy.relative_ratio_var),
                feed_dict={
                    decentralized_policy.advantage: advantage,
                    decentralized_policy.action_prob: a_p,
                    decentralized_policy.ref_reward: r,
                    decentralized_policy.reward_pre.inputs: s,
                    decentralized_policy.onehot_action: to_categorical(a, num_classes=self.argus.env.action_dim),
                    decentralized_policy.current_actor.inputs: s,
                    decentralized_policy.target_actor.inputs: s,
                    })
        else:
            actor_network_loss, _, relative_ratio_var = self.sess.run(
                self._contrib_run_object(
                    decentralized_policy.network_loss,
                    decentralized_policy.actor_train_op,
                    decentralized_policy.relative_ratio_var),
                feed_dict={
                    decentralized_policy.advantage: advantage,
                    decentralized_policy.action_prob: a_p,
                    decentralized_policy.ref_reward: r,
                    decentralized_policy.onehot_action: to_categorical(a, num_classes=self.argus.env.action_dim),
                    decentralized_policy.current_actor.inputs: s,
                })
            summary_actor_loss = []
        return actor_network_loss, summary_actor_loss, relative_ratio_var

    def _critic_train(self, s, a, a_d, r, s_, terminate, decentralized_policy, **kwargs):
        agents_mix_reward = []
        agents_pre_reward_loss = []
        if self.argus.reward.dist_reward_fit or (not self.argus.reward.dist_reward_fit and not self.argus.reward.global_reward_prediction):
            for agent_index in range(self.argus.env.agent_num):
                for reward_train_index in range(3):
                    mix_reward, pre_reward_loss, summary_pre_reward_loss, _ = self.sess.run(
                        self._contrib_run_object(
                            decentralized_policy[agent_index].mix_reward, 
                            decentralized_policy[agent_index].reward_pre_loss,
                            self.summary[self.verbose_info["reward{}_loss".format(agent_index)]["summary_index"]],
                            decentralized_policy[agent_index].reward_train_op),
                        feed_dict={
                            decentralized_policy[agent_index].reward_pre.inputs: s[:, agent_index, :],
                            decentralized_policy[agent_index].ref_reward: r[:, agent_index, :],
                            decentralized_policy[agent_index].onehot_action: to_categorical(a[:, agent_index, :], num_classes=self.argus.env.action_dim)
                        })
                agents_pre_reward_loss.append(pre_reward_loss)
                if self.argus.reward.dist_reward_fit:
                    agents_mix_reward.append(np.expand_dims(mix_reward, axis=-2))
                else:
                    agents_mix_reward.append(np.expand_dims(np.tile(mix_reward, (1, self.argus.env.action_dim)), axis=-2))
                self.summary_writer.add_summary(summary_pre_reward_loss, global_step=kwargs["time_step"])
            dre_mix_reward = np.concatenate(agents_mix_reward, axis=-2)
        else:
            agents_pre_reward, pre_reward_loss, _ = self.sess.run(
                self._contrib_run_object(
                    self.centralized_critic.pre_reward,
                    self.centralized_critic.reward_pre_loss,
                    self.centralized_critic.reward_train_op),
                feed_dict={self.centralized_critic.global_reward_pre.inputs: np.reshape(s, [-1, self.argus.reward.input_dim*self.argus.env.agent_num]),
                           self.centralized_critic.lumped_reward: r,
                           })
            dre_mix_reward = np.tile(np.expand_dims(agents_pre_reward, axis=-1), (1, self.argus.env.agent_num, self.argus.env.action_dim))
            agents_pre_reward_loss.append(pre_reward_loss)

        lumped_reward, mixed_reward = reward_aggregation_process(
            reward_aggregation_type=kwargs["argus"].reward.reward_aggregation_type, dre_mix_reward=dre_mix_reward,
            environmental_reward=r, argus=kwargs["argus"])
        advantage, critic_network_loss, summary_critic_loss, _ = self.sess.run(
            self._contrib_run_object(
                self.centralized_critic.dre_advantage,
                self.centralized_critic.critic_network_loss,
                self.summary[self.verbose_info["critic_loss"]["summary_index"]],
                self.centralized_critic.train_op),
            feed_dict={self.centralized_critic.current_critic.inputs: s,
                       self.centralized_critic.target_critic.inputs: s_,
                       self.centralized_critic.lumped_reward: lumped_reward,
                       self.centralized_critic.mixed_reward: mixed_reward,
                       self.centralized_critic.action_dist: a_d,
                       self.centralized_critic.terminate: terminate})
        return advantage, critic_network_loss, summary_critic_loss, agents_pre_reward_loss

    def _policy_train_multi_discrete(self, s, a, a_p, r, s_, advantage, decentralized_policy, summary_var, **kwargs):
        actor_network_loss, summary_actor_loss, _, relative_ratio_var = self.sess.run(
            self._contrib_run_object(
                decentralized_policy.network_loss,
                summary_var,
                decentralized_policy.actor_train_op,
                decentralized_policy.relative_ratio_var
            ),
            feed_dict={
                decentralized_policy.advantage: advantage,
                decentralized_policy.action_prob: a_p,
                decentralized_policy.ref_reward: r,
                decentralized_policy.reward_pre.inputs: s,
                decentralized_policy.onehot_action: np.concatenate(
                            [to_categorical(a[:, i], num_classes=self.argus.env.action_ncat[i]) for i in range(self.argus.env.n_action)], axis=-1),
                decentralized_policy.current_actor.inputs: s,
                })
        return actor_network_loss, summary_actor_loss, relative_ratio_var

    def _critic_train_multi_discrete(self, s, a, a_d, r, s_, terminate, decentralized_policy, **kwargs):
        agents_mix_reward = []
        agents_pre_reward_loss = []
        if self.argus.reward.dist_reward_fit or (not self.argus.reward.dist_reward_fit and not self.argus.reward.global_reward_prediction):
            for agent_index in range(self.argus.env.agent_num):
                for reward_train_index in range(3):
                    mix_reward, pre_reward_loss, summary_pre_reward_loss, _ = self.sess.run(
                        self._contrib_run_object(
                            decentralized_policy[agent_index].mix_reward,
                            decentralized_policy[agent_index].reward_pre_loss,
                            self.summary[self.verbose_info["reward{}_loss".format(agent_index)]["summary_index"]],
                            decentralized_policy[agent_index].reward_train_op),
                        feed_dict={
                            decentralized_policy[agent_index].reward_pre.inputs: s[:, agent_index, :],
                            decentralized_policy[agent_index].ref_reward: r[:, agent_index, :],
                            decentralized_policy[agent_index].onehot_action: np.concatenate(
                                [to_categorical(a[:, agent_index, i], num_classes=self.argus.env.action_ncat[i]) for i in range(self.argus.env.n_action)], axis=-1)
                        })
                agents_pre_reward_loss.append(pre_reward_loss)
                if self.argus.reward.dist_reward_fit:
                    agents_mix_reward.append(np.expand_dims(mix_reward, axis=-2))
                else:
                    agents_mix_reward.append(np.expand_dims(np.tile(mix_reward, (1, self.argus.env.action_dim)), axis=-2))
                self.summary_writer.add_summary(summary_pre_reward_loss, global_step=kwargs["time_step"])
            dre_mix_reward = np.concatenate(agents_mix_reward, axis=-2)
        else:
            agents_pre_reward, pre_reward_loss, _ = self.sess.run(
                self._contrib_run_object(
                    self.centralized_critic.pre_reward,
                    self.centralized_critic.reward_pre_loss,
                    self.centralized_critic.reward_train_op),
                feed_dict={self.centralized_critic.global_reward_pre.inputs: np.reshape(s, [-1, self.argus.reward.input_dim*self.argus.env.agent_num]),
                           self.centralized_critic.lumped_reward: r,
                           })
            dre_mix_reward = np.tile(np.expand_dims(agents_pre_reward, axis=-1), (1, self.argus.env.agent_num, self.argus.env.action_dim))
            agents_pre_reward_loss.append(pre_reward_loss)

        lumped_reward, mixed_reward = reward_aggregation_process(
            reward_aggregation_type=kwargs["argus"].reward.reward_aggregation_type, dre_mix_reward=dre_mix_reward,
            environmental_reward=r, argus=kwargs["argus"])

        advantage, critic_network_loss, summary_critic_loss, _ = self.sess.run(
            self._contrib_run_object(
                self.centralized_critic.dre_advantage,
                self.centralized_critic.critic_network_loss,
                self.summary[self.verbose_info["critic_loss"]["summary_index"]],
                self.centralized_critic.train_op),
            feed_dict={self.centralized_critic.current_critic.inputs: s,
                       self.centralized_critic.target_critic.inputs: s_,
                       self.centralized_critic.lumped_reward: lumped_reward,
                       self.centralized_critic.mixed_reward: mixed_reward,
                       self.centralized_critic.action_dist: a_d,
                       self.centralized_critic.terminate: terminate})
        return advantage, critic_network_loss, summary_critic_loss, agents_pre_reward_loss




