
import tensorflow._api.v2.compat.v1 as tf
from .base_model import base_model
from .graph_head import GAT_head
from .reward_model import global_reward_pre

class ma_critic(base_model):
    def __init__(self, model_name, argus, father_scope, is_trainable=True):
        super(ma_critic, self).__init__()
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
                shape=[None, self.argus.env.agent_num, self.argus.critic.input_dim], dtype=tf.float32, name="{}_INPUT".format(self.model_name))
            if self.father_scope is not None:
                self.gat_out = GAT_head(model_name="GAT_head", argus=self.argus, inputs=self.inputs,
                                        father_scope=self.father_scope + "/{}".format(self.model_name),
                                        is_trainable=self.is_trainable)
            else:
                self.gat_out = GAT_head(model_name="GAT_head", argus=self.argus, inputs=self.inputs,
                                        father_scope=self.model_name,
                                        is_trainable=self.is_trainable)

            for layer_index in range(2):
                if layer_index == 0:
                    self.layers.append(
                        tf.layers.dense(self.gat_out.outputs, 64, activation=tf.nn.leaky_relu,
                                        trainable=self.is_trainable, name="{}_layer_{}".format(self.model_name, layer_index)))
                else:
                    self.layers.append(
                        tf.layers.dense(self.layers[layer_index-1], 64, activation=tf.nn.leaky_relu,
                                        trainable=self.is_trainable, name="{}_layer_{}".format(self.model_name, layer_index)))
            self.outputs = tf.layers.dense(self.layers[-1], 1, activation=None, name="{}_OUTPUT".format(self.model_name),
                                           trainable=self.is_trainable)

        if self.father_scope is not None:
            self.variables = self.collect_variables(scope=self.father_scope + "/{}".format(self.model_name))
            self.bn_train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.father_scope + "/" + self.model_name)
        else:
            self.variables = self.collect_variables(scope=self.model_name)
            self.bn_train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_name)

class centralized_critic(base_model):
    def __init__(self, model_name, argus, father_scope=None, is_trainable=True):
        super(centralized_critic, self).__init__()
        self.model_name = model_name
        self.argus = argus
        self.father_scope = father_scope
        self.is_trainable = is_trainable
        self.build()

    def _build(self):
        with tf.variable_scope(self.model_name):
            if self.father_scope is not None:
                self.current_critic = ma_critic(
                    model_name="current_centralized_critic", argus=self.argus,
                    father_scope=self.father_scope + "/{}".format(self.model_name), is_trainable=True)
                self.target_critic = ma_critic(
                    model_name="target_centralized_critic", argus=self.argus,
                    father_scope=self.father_scope + "/{}".format(self.model_name), is_trainable=False)
                if (not self.argus.reward.dist_reward_fit) and self.argus.reward.global_reward_prediction:
                    self.global_reward_pre = global_reward_pre(
                        model_name="global_reward_pred", argus=self.argus,
                        father_scope=self.father_scope + "/{}".format(self.model_name), is_trainable=True)
            else:
                self.current_critic = ma_critic(
                    model_name="current_centralized_critic", argus=self.argus,
                    father_scope=self.model_name, is_trainable=True)
                self.target_critic = ma_critic(
                    model_name="target_centralized_critic", argus=self.argus,
                    father_scope=self.model_name, is_trainable=False)
                if (not self.argus.reward.dist_reward_fit) and self.argus.reward.global_reward_prediction:
                    self.global_reward_pre = global_reward_pre(
                        model_name="global_reward_pred", argus=self.argus, father_scope=self.model_name, is_trainable=True)

            self.saver = tf.train.Saver(
                max_to_keep=self.argus.save.total_model_num, var_list=self.current_critic.variables)

            with tf.variable_scope('centralized_critic_loss'):
                self.lumped_reward = tf.placeholder(shape=[None, self.argus.env.agent_num, 1], dtype=tf.float32, name='lumped_reward')
                self.terminate = tf.placeholder(shape=[None, self.argus.env.agent_num, 1], dtype=tf.float32, name='terminate')
                self.mixed_reward = tf.placeholder(shape=[None, self.argus.env.agent_num, self.argus.env.action_dim], dtype=tf.float32, name='mixed_reward')
                self.action_prob = tf.placeholder(shape=[None, self.argus.env.agent_num, 1], dtype=tf.float32, name='action_prob')
                self.action_dist = tf.placeholder(shape=[None, self.argus.env.agent_num, self.argus.env.action_dim], dtype=tf.float32, name='action_dist')
                if self.argus.env.action_type == "discrete":
                    self.policy_weighted_mixed_reward = tf.reduce_sum(self.action_dist * self.mixed_reward, axis=-1, keepdims=True)
                else:
                    multi_rewards = tf.split(self.mixed_reward, self.argus.env.action_ncat, axis=-1)
                    multi_action_dist = tf.split(self.action_dist, self.argus.env.action_ncat, axis=-1)
                    self.policy_weighted_mixed_reward = tf.reduce_mean(tf.concat(
                        [tf.reduce_sum(multi_action_dist[i] * multi_rewards[i], axis=-1, keepdims=True) for i in range(self.argus.env.n_action)], axis=-1), axis=-1, keepdims=True)
                self.discount_target_critic_out = tf.pow(self.argus.train.gamma, self.argus.train.TD) * self.target_critic.outputs
                self.current_V = self.current_critic.outputs

                self.target_V = self.policy_weighted_mixed_reward + self.discount_target_critic_out
                self.dre_Q_value = self.lumped_reward + self.discount_target_critic_out
                self.dre_advantage = self.dre_Q_value - self.current_V

                self.MSE_loss = tf.reduce_mean(tf.square(self.target_V - self.current_V))
                self.critic_network_loss = tf.cast(self.MSE_loss, dtype=tf.float32)

                if self.argus.train.batch_norm:
                    with tf.control_dependencies(self.current_critic.bn_train_ops):
                        self.critic_optimizer = tf.train.AdamOptimizer(self.argus.critic.lr)
                        self.critic_grads = self.critic_optimizer.compute_gradients(
                            loss=self.critic_network_loss, var_list=self.current_critic.variables)
                        for i, (g, v) in enumerate(self.critic_grads):
                            if g is not None:
                                self.critic_grads[i] = (tf.clip_by_value(g, -3.0, 3.0), v)
                        self.train_op = self.critic_optimizer.apply_gradients(self.critic_grads)
                else:
                    self.critic_optimizer = tf.train.AdamOptimizer(self.argus.critic.lr)
                    self.critic_grads = self.critic_optimizer.compute_gradients(
                        loss=self.critic_network_loss, var_list=self.current_critic.variables)
                    for i, (g, v) in enumerate(self.critic_grads):
                        if g is not None:
                            self.critic_grads[i] = (tf.clip_by_value(g, -3.0, 3.0), v)
                    self.train_op = self.critic_optimizer.apply_gradients(self.critic_grads)
                # self.saver = tf.train.Saver(max_to_keep=argus.total_model_num, var_list=self.current_critic.critic_paras)
                self.reload_paras = \
                    [tf.assign(oldp, p) for p, oldp in zip(self.current_critic.variables, self.target_critic.variables)]
                self.update_paras = \
                    [oldp.assign(p * self.argus.train.tau + oldp * (1-self.argus.train.tau))
                     for p, oldp in zip(self.current_critic.variables, self.target_critic.variables)]

            if not self.argus.reward.dist_reward_fit and self.argus.reward.global_reward_prediction:
                with tf.variable_scope('global_reward_pre_loss'):
                    self.pre_reward = self.global_reward_pre.outputs
                    if self.argus.env.action_type == "discrete":
                        self.mix_reward = self.global_reward_pre.outputs
                        self.reward_pre_loss = tf.reduce_mean(tf.square(self.pre_reward - tf.reduce_mean(self.lumped_reward, axis=-2)))
                    else:
                        self.mix_reward = self.global_reward_pre.outputs
                        self.reward_pre_loss = tf.reduce_mean(tf.square(self.pre_reward - tf.reduce_mean(self.lumped_reward, axis=-2)))
                self.reward_pre_optimizer = tf.train.AdamOptimizer(self.argus.reward.lr)
                self.reward_pre_grads = self.reward_pre_optimizer.compute_gradients(
                    loss=self.reward_pre_loss, var_list=self.global_reward_pre.variables)
                for i, (g, v) in enumerate(self.reward_pre_grads):
                    if g is not None:
                        self.reward_pre_grads[i] = (tf.clip_by_value(g, -3.0, 3.0), v)
                self.reward_train_op = self.reward_pre_optimizer.apply_gradients(self.reward_pre_grads)

    def _update_parameters(self, sess):
        sess.run(self.update_paras)

    def _reload_parameters(self, sess):
        sess.run(self.reload_paras)
