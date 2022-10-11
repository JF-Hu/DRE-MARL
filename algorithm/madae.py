

import tensorflow._api.v2.compat.v1 as tf
from DAE.algorithm.hyperpara_setting import get_variant_config
from DAE.models.MADAE_model import MADAE

algorithm, argus = get_variant_config()
madae = MADAE(algorithm, argus)
summary_writer = tf.summary.FileWriter("./logs", graph=madae.graph)


















