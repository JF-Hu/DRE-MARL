
import tensorflow._api.v2.compat.v1 as tf
from .base_model import base_model
import tensorflow_probability as tfp
from env_utils import (ConditionalShift, ConditionalScale)

class reward_pre(base_model):
    def __init__(self, model_name, argus, father_scope=None, is_trainable=True):
        super(reward_pre, self).__init__()
        self.model_name = model_name
        self.argus = argus
        self.father_scope = father_scope
        self.is_trainable = is_trainable
        self.base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(argus.env.action_dim),
            scale_diag=tf.ones(argus.env.action_dim))
        self.raw_action_distribution = tfp.bijectors.Chain((
            ConditionalShift(name='shift'),
            ConditionalScale(name='scale'),
        ))(self.base_distribution)
        self._action_post_processor = {
            True: tfp.bijectors.Tanh(),
            False: tfp.bijectors.Identity(),
        }[True]
        self.action_distribution = self._action_post_processor(
            self.raw_action_distribution)
        if self.argus.reward.dist_reward_fit:
            self.build()
        else:
            self.p2p_reward_predict_build()

    def contribute_loss(self):
        pass

    def collect_variables(self, scope, only_trainable_variable=False):
        if only_trainable_variable:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def contribute_summary(self, sess, summary_path, summary_variable_list, variable_name_list):
        pass


    def _build(self):
        with tf.variable_scope(self.model_name):
            self.inputs = tf.placeholder(
                shape=[None, self.argus.reward.input_dim], dtype=tf.float32, name=self.model_name + "_INPUT")

            for layer_index in range(2):
                if layer_index == 0:
                    self.layers.append(
                        tf.layers.dense(self.inputs, 64, activation=tf.nn.leaky_relu,
                                        trainable=self.is_trainable, name="{}_layer_{}".format(self.model_name, layer_index)))
                else:
                    self.layers.append(
                        tf.layers.dense(self.layers[layer_index-1], 64, activation=tf.nn.leaky_relu,
                                        trainable=self.is_trainable, name="{}_layer_{}".format(self.model_name, layer_index)))
            self.layers_outputs = tf.layers.dense(self.layers[-1], self.argus.env.action_dim*2, "linear", name="{}_layers_outputs".format(self.model_name))
            self.shift, scale = tf.split(self.layers_outputs, num_or_size_splits=2, axis=-1)
            self.scale = tf.math.softplus(scale, name="{}_scale_softplus_op".format(self.model_name)) + 1e-5
            self.outputs = tf.concat([self.shift, self.scale], axis=-1, name="{}_OUTPUT".format(self.model_name))
            if self.father_scope is not None:
                self.variables = self.collect_variables(scope=self.father_scope + "/{}".format(self.model_name))
            else:
                self.variables = self.collect_variables(scope=self.model_name)

    def p2p_reward_predict_build(self):
        with tf.variable_scope(self.model_name):
            self.inputs = tf.placeholder(
                shape=[None, self.argus.reward.input_dim], dtype=tf.float32, name=self.model_name + "_INPUT")

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
                self.outputs = tf.layers.dense(self.layers[-1], 1, "linear", name="{}_OUTPUT".format(self.model_name))
            else:
                self.outputs = tf.layers.dense(self.layers[-1], 1, "linear", name="{}_OUTPUT".format(self.model_name))
            if self.father_scope is not None:
                self.variables = self.collect_variables(scope=self.father_scope + "/{}".format(self.model_name))
            else:
                self.variables = self.collect_variables(scope=self.model_name)




class global_reward_pre(base_model):
    def __init__(self, model_name, argus, father_scope=None, is_trainable=True):
        super(global_reward_pre, self).__init__()
        self.model_name = model_name
        self.argus = argus
        self.father_scope = father_scope
        self.is_trainable = is_trainable
        self.base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(argus.env.action_dim),
            scale_diag=tf.ones(argus.env.action_dim))
        self.raw_action_distribution = tfp.bijectors.Chain((
            ConditionalShift(name='shift'),
            ConditionalScale(name='scale'),
        ))(self.base_distribution)
        self._action_post_processor = {
            True: tfp.bijectors.Tanh(),
            False: tfp.bijectors.Identity(),
        }[True]
        self.action_distribution = self._action_post_processor(
            self.raw_action_distribution)
        if self.argus.reward.dist_reward_fit:
            self.build()
        else:
            self.p2p_reward_predict_build()

    def contribute_loss(self):
        pass

    def collect_variables(self, scope, only_trainable_variable=False):
        if only_trainable_variable:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def contribute_summary(self, sess, summary_path, summary_variable_list, variable_name_list):
        pass

    def _build(self):
        with tf.variable_scope(self.model_name):
            self.inputs = tf.placeholder(
                shape=[None, self.argus.reward.input_dim*self.argus.reward.agent_num], dtype=tf.float32, name=self.model_name + "_INPUT")

            for layer_index in range(2):
                if layer_index == 0:
                    self.layers.append(
                        tf.layers.dense(self.inputs, 64, activation=tf.nn.leaky_relu,
                                        trainable=self.is_trainable, name="{}_layer_{}".format(self.model_name, layer_index)))
                else:
                    self.layers.append(
                        tf.layers.dense(self.layers[layer_index-1], 64, activation=tf.nn.leaky_relu,
                                        trainable=self.is_trainable, name="{}_layer_{}".format(self.model_name, layer_index)))
            self.layers_outputs = tf.layers.dense(self.layers[-1], self.argus.env.action_dim*2, "linear", name="{}_layers_outputs".format(self.model_name))
            self.shift, scale = tf.split(self.layers_outputs, num_or_size_splits=2, axis=-1)
            self.scale = tf.math.softplus(scale, name="{}_scale_softplus_op".format(self.model_name)) + 1e-5
            self.outputs = tf.concat([self.shift, self.scale], axis=-1, name="{}_OUTPUT".format(self.model_name))
            if self.father_scope is not None:
                self.variables = self.collect_variables(scope=self.father_scope + "/{}".format(self.model_name))
            else:
                self.variables = self.collect_variables(scope=self.model_name)

    def p2p_reward_predict_build(self):
        with tf.variable_scope(self.model_name):
            self.inputs = tf.placeholder(
                shape=[None, self.argus.reward.input_dim*self.argus.env.agent_num], dtype=tf.float32, name=self.model_name + "_INPUT")

            for layer_index in range(2):
                if layer_index == 0:
                    self.layers.append(
                        tf.layers.dense(self.inputs, 64, activation=tf.nn.leaky_relu,
                                        trainable=self.is_trainable, name="{}_layer_{}".format(self.model_name, layer_index)))
                else:
                    self.layers.append(
                        tf.layers.dense(self.layers[layer_index-1], 64, activation=tf.nn.leaky_relu,
                                        trainable=self.is_trainable, name="{}_layer_{}".format(self.model_name, layer_index)))
            self.outputs = tf.layers.dense(self.layers[-1], 1, "linear", name="{}_OUTPUT".format(self.model_name))
            if self.father_scope is not None:
                self.variables = self.collect_variables(scope=self.father_scope + "/{}".format(self.model_name))
            else:
                self.variables = self.collect_variables(scope=self.model_name)
