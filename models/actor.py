import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from .base_model import base_model
from .reward_model import reward_pre

class ma_actor(base_model):
    def __init__(self, model_name, argus, father_scope, is_trainable=True):
        super(ma_actor, self).__init__()
        self.model_name = model_name
        self.argus = argus
        self.father_scope = father_scope
        self.is_trainable = is_trainable
        self.build()

    def collect_variables(self, scope, only_trainable_variable=False):
        if only_trainable_variable:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def _build(self):
        with tf.variable_scope(self.model_name):
            self.inputs = tf.placeholder(
                shape=[None, self.argus.actor.input_dim], dtype=tf.float32, name="{}_INPUT".format(self.model_name))

            for layer_index in range(2):
                if layer_index == 0:
                    self.layers.append(
                        tf.layers.dense(self.inputs, 64, activation=tf.nn.leaky_relu,
                                        trainable=self.is_trainable, name="{}_layer_{}".format(self.model_name, layer_index)))
                else:
                    self.layers.append(
                        tf.layers.dense(self.layers[layer_index-1], 64, activation=tf.nn.leaky_relu,
                                        trainable=self.is_trainable, name="{}_layer_{}".format(self.model_name, layer_index)))
            if self.argus.env.action_type == "discrete":
                self.outputs = tf.layers.dense(self.layers[-1], self.argus.env.action_dim, activation=tf.nn.softmax, name="{}_OUTPUT".format(self.model_name),
                                               trainable=self.is_trainable)
            elif self.argus.env.action_type == "multi_discrete":
                self.actions = tf.layers.dense(self.layers[-1], self.argus.env.action_dim, activation=None, name="{}_OUTPUT".format(self.model_name), trainable=self.is_trainable)
                self.softmax_actions = tf.split(self.actions, self.argus.env.action_ncat, axis=-1)
                self.outputs = tf.concat([tf.nn.softmax(softmax_action, axis=-1) for softmax_action in self.softmax_actions], axis=-1)

        if self.father_scope is not None:
            self.variables = self.collect_variables(scope=self.father_scope + "/{}".format(self.model_name))
            self.bn_train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.father_scope + "/" + self.model_name)
        else:
            self.variables = self.collect_variables(scope=self.model_name)
            self.bn_train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_name)

class ma_policy(base_model):
    def __init__(self, model_name, argus, father_scope, is_trainable=True):
        super(ma_policy, self).__init__()
        self.model_name = model_name
        self.argus = argus
        self.father_scope = father_scope
        self.is_trainable = is_trainable
        self.build()

    def collect_variables(self, scope, only_trainable_variable=False):
        if only_trainable_variable:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def _build(self):
        with tf.variable_scope(self.model_name):
            if self.father_scope is not None:
                self.current_actor = ma_actor(
                    model_name="current_actor", argus=self.argus,
                    father_scope=self.father_scope + "/{}".format(self.model_name), is_trainable=True)
                self.target_actor = ma_actor(
                    model_name="target_actor", argus=self.argus,
                    father_scope=self.father_scope + "/{}".format(self.model_name), is_trainable=False)
                self.reward_pre = reward_pre(
                    model_name="reward_pred", argus=self.argus,
                    father_scope=self.father_scope + "/{}".format(self.model_name), is_trainable=True)

            else:
                self.current_actor = ma_actor(
                    model_name="current_actor", argus=self.argus, father_scope=self.model_name, is_trainable=True)
                self.target_actor = ma_actor(
                    model_name="target_actor", argus=self.argus, father_scope=self.model_name, is_trainable=False)
                self.reward_pre = reward_pre(
                    model_name="reward_pred", argus=self.argus, father_scope=self.model_name, is_trainable=True)

            with tf.variable_scope('policy_loss'):
                self.ref_reward = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='ref_reward')
                self.advantage = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantage')
                self.action_prob = tf.placeholder(shape=[None, self.argus.env.n_action], dtype=tf.float32, name='action_prob')
                self.onehot_action = tf.placeholder(shape=[None, self.argus.env.action_dim], dtype=tf.float32, name='onehot_action')
                self.pi_old = tf.reshape(
                    tf.reduce_sum(self.target_actor.outputs * self.onehot_action, axis=-1, keepdims=True), [-1, 1])
                if self.argus.env.action_type == "discrete":
                    self.pi = tf.reshape(
                        tf.reduce_sum(self.current_actor.outputs * self.onehot_action, axis=-1, keepdims=True), [-1, 1])
                    self.pi_entropy = tf.reduce_sum(
                        -self.current_actor.outputs * tf.log(self.current_actor.outputs + self.argus.train.delta), axis=-1, keepdims=True)
                    self.ratio = (self.pi + 1e-5) / (self.action_prob + 1e-5)
                    self.advantage_flag = self.advantage / tf.abs(self.advantage)
                    self.ratio_normal = tf.minimum(self.ratio * self.advantage_flag, tf.clip_by_value(self.ratio, 1 - self.argus.train.epsilon, 1 + self.argus.train.epsilon) * self.advantage_flag)
                    self.ratio_madre = tf.minimum(self.ratio,tf.clip_by_value(self.ratio, 1 - self.argus.train.epsilon, 1 + self.argus.train.epsilon))
                    self.relative_ratio_var = (tf.reduce_mean(tf.square(self.ratio_normal)) - tf.reduce_mean(tf.square(self.ratio_madre)))/tf.reduce_mean(tf.square(self.ratio_madre))

                    self.network_loss = tf.reduce_mean(
                        tf.minimum(self.ratio, tf.clip_by_value(
                            self.ratio, 1 - self.argus.train.epsilon, 1 + self.argus.train.epsilon)) * self.advantage + 0.3 * self.pi_entropy)
                else:
                    multi_pis = tf.split(self.current_actor.outputs * self.onehot_action, self.argus.env.action_ncat, axis=-1)
                    multi_outputs = tf.split(self.current_actor.outputs, self.argus.env.action_ncat, axis=-1)
                    self.pi = tf.reshape(
                        tf.concat([tf.reduce_sum(multi_pis[i], axis=-1, keepdims=True) for i in range(self.argus.env.n_action)], axis=-1), [-1, self.argus.env.n_action])
                    self.pi_entropy = tf.concat([tf.reduce_sum(-multi_outputs[i] * tf.log(multi_outputs[i] + self.argus.train.delta), axis=-1, keepdims=True) for i in range(self.argus.env.n_action)], axis=-1)
                    self.ratio = (self.pi + 1e-5) / (self.action_prob + 1e-5)

                    self.advantage_flag = self.advantage / tf.abs(self.advantage)
                    self.ratio_normal = tf.minimum(self.ratio * self.advantage_flag, tf.clip_by_value(self.ratio, 1 - self.argus.train.epsilon, 1 + self.argus.train.epsilon) * self.advantage_flag)
                    self.ratio_madre = tf.minimum(self.ratio, tf.clip_by_value(self.ratio, 1 - self.argus.train.epsilon, 1 + self.argus.train.epsilon))
                    self.relative_ratio_var = tf.reduce_mean(
                        (tf.reduce_mean(tf.square(self.ratio_normal), axis=0, keepdims=True) - tf.reduce_mean(
                            tf.square(self.ratio_madre), axis=0, keepdims=True)) / tf.reduce_mean(
                            tf.square(self.ratio_madre), axis=0, keepdims=True)
                    )

                    self.network_loss = tf.reduce_mean(tf.reduce_mean(
                        tf.minimum(self.ratio, tf.clip_by_value(
                            self.ratio, 1 - self.argus.train.epsilon, 1 + self.argus.train.epsilon)) * self.advantage + 0.3 * self.pi_entropy, axis=0, keepdims=True))
                if self.argus.train.batch_norm is True:
                    with tf.control_dependencies(self.current_actor.bn_train_ops):
                        self.actor_optimizer = tf.train.AdamOptimizer(self.argus.actor.lr)
                        self.actor_grads = self.actor_optimizer.compute_gradients(
                            loss=-self.network_loss, var_list=self.current_actor.variables)
                        for i, (g, v) in enumerate(self.actor_grads):
                            if g is not None:
                                self.actor_grads[i] = (tf.clip_by_value(g, -3.0, 3.0), v)
                        self.actor_train_op = self.actor_optimizer.apply_gradients(self.actor_grads)
                else:
                    self.actor_optimizer = tf.train.AdamOptimizer(self.argus.actor.lr)
                    self.actor_grads = self.actor_optimizer.compute_gradients(
                        loss=-self.network_loss, var_list=self.current_actor.variables)
                    for i, (g, v) in enumerate(self.actor_grads):
                        if g is not None:
                            self.actor_grads[i] = (tf.clip_by_value(g, -3.0, 3.0), v)
                    self.actor_train_op = self.actor_optimizer.apply_gradients(self.actor_grads)
                self.reload_paras = \
                    [tf.assign(oldp, p) for p, oldp in zip(self.current_actor.variables, self.target_actor.variables)]
                self.update_paras = \
                    [oldp.assign(p * self.argus.train.tau + oldp * (1 - self.argus.train.tau))
                     for p, oldp in zip(self.current_actor.variables, self.target_actor.variables)]

            with tf.variable_scope('reward_pre_loss'):
                if self.argus.reward.dist_reward_fit:
                    self.pre_reward = self.reward_pre.shift
                    if self.argus.env.action_type == "discrete":
                        self.masked_shift = tf.reduce_sum(self.reward_pre.shift * self.onehot_action, axis=-1 ,keepdims=True)
                        self.masked_scale = tf.reduce_sum(self.reward_pre.scale * self.onehot_action, axis=-1 ,keepdims=True)
                        self.mix_reward = self.pre_reward * (1.0 - self.onehot_action) + tf.tile(self.ref_reward, [1, self.argus.env.action_dim]) * self.onehot_action
                        self.mean_pre_reward = tf.clip_by_value(
                            tf.reshape(tf.square(tf.reduce_mean(self.reward_pre.shift, axis=-1, keepdims=True) - self.ref_reward), [-1, 1]), clip_value_min=0.0, clip_value_max=1000.0)
                        self.var_norm_loss = tf.norm(self.masked_scale, ord=2, axis=-1, keepdims=True)
                        self.pre_reward_mean, self.pre_reward_variance = tf.nn.moments(self.pre_reward, axes=-1, keepdims=True)
                        self.vector_reward_pre_loss = (0.5 * tf.log(2 * np.pi) + tf.log(self.masked_scale + self.argus.train.delta) + tf.square(
                            (self.ref_reward - self.masked_shift) / self.masked_scale) / 2.0) + 0.1 * self.var_norm_loss + 10.0*self.pre_reward_variance
                        self.reward_pre_loss = tf.reduce_mean(self.vector_reward_pre_loss)
                    else:
                        multi_shifts = tf.split(self.reward_pre.shift, self.argus.env.action_ncat, axis=-1)
                        multi_shift_actions = tf.split(self.reward_pre.shift * self.onehot_action, self.argus.env.action_ncat, axis=-1)
                        multi_scale_actions = tf.split(self.reward_pre.scale * self.onehot_action, self.argus.env.action_ncat, axis=-1)
                        self.masked_shift = tf.concat([tf.reduce_sum(multi_shift_actions[i], axis=-1 ,keepdims=True) for i in range(self.argus.env.n_action)], axis=-1)
                        self.masked_scale = tf.concat([tf.reduce_sum(multi_scale_actions[i], axis=-1 ,keepdims=True) for i in range(self.argus.env.n_action)], axis=-1)
                        self.mix_reward = self.pre_reward * (1.0 - self.onehot_action) + tf.tile(self.ref_reward, [1, self.argus.env.action_dim]) * self.onehot_action
                        self.mean_pre_reward = tf.clip_by_value(
                            tf.concat([tf.reduce_mean(multi_shifts[i], axis=-1, keepdims=True) for i in range(self.argus.env.n_action)], axis=-1) - self.ref_reward, clip_value_min=0.0, clip_value_max=1000.0)
                        self.var_norm_loss = tf.concat([tf.norm(tf.slice(self.masked_scale, [0, 0], [-1, i]), ord=2, axis=-1, keepdims=True) for i in range(self.argus.env.n_action)], axis=-1)
                        self.pre_reward_mean, self.pre_reward_variance = tf.nn.moments(self.pre_reward, axes=-1, keepdims=True)
                        self.vector_reward_pre_loss = (0.5*tf.log(2*np.pi) + tf.log(self.masked_scale + self.argus.train.delta) + tf.square((self.ref_reward - self.masked_shift) / self.masked_scale) / 2.0) + \
                                                      0.1*self.var_norm_loss + 10.0*self.pre_reward_variance
                        self.reward_pre_loss = tf.reduce_mean(tf.reduce_mean(self.vector_reward_pre_loss, axis=-1, keepdims=True))
                else:
                    self.pre_reward = self.reward_pre.outputs
                    if self.argus.env.action_type == "discrete":
                        self.mix_reward = self.reward_pre.outputs
                        self.reward_pre_loss = tf.reduce_mean(tf.square(self.pre_reward - self.ref_reward))
                    else:
                        self.mix_reward = self.reward_pre.outputs
                        self.reward_pre_loss = tf.reduce_mean(tf.square(self.pre_reward - self.ref_reward))
                self.reward_pre_optimizer = tf.train.AdamOptimizer(self.argus.reward.lr)
                self.reward_pre_grads = self.reward_pre_optimizer.compute_gradients(
                    loss=self.reward_pre_loss, var_list=self.reward_pre.variables)
                for i, (g, v) in enumerate(self.reward_pre_grads):
                    if g is not None:
                        self.reward_pre_grads[i] = (tf.clip_by_value(g, -3.0, 3.0), v)
                self.reward_train_op = self.reward_pre_optimizer.apply_gradients(self.reward_pre_grads)


class decentralized_actor(base_model):
    def __init__(self, model_name, argus, father_scope=None, is_trainable=True):
        super(decentralized_actor, self).__init__()
        self.model_name = model_name
        self.argus = argus
        self.father_scope = father_scope
        self.is_trainable = is_trainable
        self.policies = []
        self.build()

    def _build(self):
        with tf.variable_scope(self.model_name):
            for agent_index in range(self.argus.env.agent_num):
                if self.father_scope is not None:
                    self.policies.append(
                        ma_policy(model_name="policy_{}".format(agent_index), argus=self.argus,
                                  father_scope=self.father_scope + "/{}".format(self.model_name)))
                else:
                    self.policies.append(
                        ma_policy(model_name="policy_{}".format(agent_index), argus=self.argus,
                                  father_scope=self.model_name))
        self.total_variable = self.policies[0].current_actor.variables
        for i in range(len(self.policies)-1):
            self.total_variable = self.total_variable + self.policies[i+1].current_actor.variables
        self.saver = tf.train.Saver(
            max_to_keep=self.argus.save.total_model_num,
            var_list=self.total_variable)

    def _update_parameters(self, sess, agent_index):
        sess.run(self.policies[agent_index].update_paras)

    def _reload_parameters(self, sess, agent_index):
        sess.run(self.policies[agent_index].reload_paras)

    def reNormalize(self, argus, input_vector, mask_vector=None):
        if mask_vector is not None:
            actionExpSum = tf.reduce_sum(tf.exp(input_vector * mask_vector) * mask_vector, axis=-1, keepdims=True)
            actionMasked = (tf.exp(input_vector * mask_vector) * mask_vector) / actionExpSum
            return tf.reshape(actionMasked, [-1, argus.env.action_dim])
        else:
            return input_vector