
from .base_model import base_model
import tensorflow._api.v2.compat.v1 as tf

class Attn_head(base_model):
    def __init__(self, model_name, argus, father_scope, is_trainable, inputs):
        super(Attn_head, self).__init__()
        self.model_name = model_name
        self.argus = argus
        self.father_scope = father_scope
        self.is_trainable = is_trainable
        self.inputs = inputs
        self.build()

    def collect_variables(self, scope, only_trainable_variable=False):
        if only_trainable_variable:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def _build(self):
        with tf.variable_scope(self.model_name):
            inputs = tf.layers.dense(self.inputs, self.argus.graph.Con1d_kernel_num, use_bias=False,
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     trainable=self.is_trainable)
            if self.argus.graph.in_drop != 0.0:
                inputs = tf.nn.dropout(inputs, 1.0 - self.argus.graph.in_drop)
            if self.argus.graph.in_norm:
                inputs = tf.reshape(
                    inputs / tf.norm(inputs, ord=2, axis=-1, keepdims=True), [-1, inputs.shape[1], inputs.shape[2]])
            self.inputs_Q = tf.layers.dense(inputs, self.argus.graph.Con1d_kernel_num, use_bias=False,
                                             kernel_initializer=tf.glorot_uniform_initializer(),
                                             trainable=self.is_trainable)
            self.inputs_K = tf.layers.dense(inputs, self.argus.graph.Con1d_kernel_num, use_bias=False,
                                             kernel_initializer=tf.glorot_uniform_initializer(),
                                             trainable=self.is_trainable)
            self.h_i = tf.reshape(tf.tile(
                input=tf.reshape(self.inputs_Q, [-1, self.argus.env.agent_num, self.argus.graph.Con1d_kernel_num]), multiples=[1, 1, self.argus.env.agent_num]),
                [-1, self.argus.env.agent_num * self.argus.env.agent_num, self.argus.graph.Con1d_kernel_num])
            self.h_j = tf.reshape(tf.tile(
                input=tf.reshape(self.inputs_K, [-1, self.argus.env.agent_num, self.argus.graph.Con1d_kernel_num]), multiples=[1, self.argus.env.agent_num, 1]),
                [-1, self.argus.env.agent_num * self.argus.env.agent_num, self.argus.graph.Con1d_kernel_num])
            self.a_input = tf.concat([self.h_i, self.h_j], axis=-1)
            self.e = tf.layers.dense(self.a_input, 1, use_bias=False, activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.glorot_uniform_initializer(), trainable=self.is_trainable)
            self.atten_weight = tf.nn.softmax(tf.reshape(self.e, [-1, self.argus.env.agent_num, self.argus.env.agent_num]), axis=-1, name="atten_weight")

            if self.argus.graph.coef_drop != 0.0:
                self.atten_weight = tf.nn.dropout(self.atten_weight, 1.0 - self.argus.graph.coef_drop)
            if self.argus.graph.in_drop != 0.0:
                inputs = tf.nn.dropout(inputs, 1.0 - self.argus.graph.in_drop)
            self.vals = tf.matmul(self.atten_weight, inputs)

            # residual connection
            if self.argus.graph.residual:
                if inputs.shape[-1] != self.vals.shape[-1]:
                    self.outputs = tf.nn.elu(self.vals + tf.layers.conv1d(
                        inputs, self.vals.shape[-1], 1, activation=tf.nn.leaky_relu,
                        kernel_initializer=tf.glorot_uniform_initializer(), trainable=self.is_trainable))
                    # raise Exception("The shape of inputs and vals are wrong!!!")
                else:
                    self.outputs = tf.nn.elu(tf.concat([self.vals, inputs], axis=-1))
            else:
                self.outputs = tf.nn.elu(self.vals)

        if self.father_scope is not None:
            self.variables = self.collect_variables(scope=self.father_scope + "/{}".format(self.model_name))
        else:
            self.variables = self.collect_variables(scope=self.model_name)

class Multi_head_atten(base_model):
    def __init__(self, model_name, argus, inputs, father_scope=None, is_trainable=True, is_output=False):
        super(Multi_head_atten, self).__init__()
        self.model_name = model_name
        self.argus = argus
        self.father_scope = father_scope
        self.is_trainable = is_trainable
        self.inputs = inputs
        self.is_output = is_output
        self.build()

    def collect_variables(self, scope, only_trainable_variable=False):
        if only_trainable_variable:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def _build(self):
        with tf.variable_scope(self.model_name):
            if self.is_output:
                self.outputs = tf.layers.dense(
                    self.inputs, 64, activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.glorot_uniform_initializer(), trainable=self.is_trainable,
                    name="gat_out_feature")
            else:
                self.atten_head_list = []
                self.atten_head_output_list = []
                for atten_index in range(self.argus.graph.atten_head_num):
                    self.atten_head_list.append(Attn_head(
                        model_name="atten_{}".format(atten_index + 1), argus=self.argus, inputs=self.inputs, father_scope=self.father_scope, is_trainable=self.is_trainable))
                    self.atten_head_output_list.append(self.atten_head_list[atten_index].outputs)
                self.outputs = tf.concat(self.atten_head_output_list, axis=-1)

        if self.father_scope is not None:
            self.variables = self.collect_variables(scope=self.father_scope + "/{}".format(self.model_name))
        else:
            self.variables = self.collect_variables(scope=self.model_name)

class GAT_head(base_model):
    def __init__(self, model_name, argus, inputs, father_scope=None, is_trainable=True):
        super(GAT_head, self).__init__()
        self.model_name = model_name
        self.argus = argus
        self.father_scope = father_scope
        self.is_trainable = is_trainable
        self.inputs = inputs
        self.build()

    def collect_variables(self, scope, only_trainable_variable=False):
        if only_trainable_variable:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def _build(self):
        with tf.variable_scope(self.model_name):
            self.input_multi_head = Multi_head_atten(
                model_name="Input_multi_head_atten", argus=self.argus, inputs=self.inputs,
                is_output=False, father_scope=self.father_scope, is_trainable=self.is_trainable)

            self.output_multi_head = Multi_head_atten(
                model_name="Output_multi_head_atten", argus=self.argus, inputs=self.input_multi_head.outputs,
                is_output=True, father_scope=self.father_scope, is_trainable=self.is_trainable)
            self.outputs = self.output_multi_head.outputs

        if self.father_scope is not None:
            self.variables = self.collect_variables(scope=self.father_scope + "/{}".format(self.model_name))
        else:
            self.variables = self.collect_variables(scope=self.model_name)





