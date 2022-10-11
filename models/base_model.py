

class base_model(object):
    def __init__(self):
        self.argus = None
        self.model_name = None

        self.saver = None
        self.father_scope = None
        self.trainable_variables = None
        self.global_variables = None

        self.inputs = None
        self.outputs = None
        self.layers = []

        self.summary = []
        self.summary_writer = []

        self.loss = []
        self.optimizer = None
        self.opt_op = None

        self.sess = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        self._build()

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def contribute_summary(self, sess, summary_path, summary_variable_list, variable_name_list):
        raise NotImplementedError

    def save(self, save_path, global_step, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        self.saver.save(sess, save_path, global_step=global_step)
        print("Model saved in file: %s" % save_path)

    def load(self, model_path, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        self.saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)