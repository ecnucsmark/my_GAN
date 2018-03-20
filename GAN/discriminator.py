import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

class Discriminator(object):

    def __init__(self):
        self.name = 'discriminator'

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            print(x)
            fc1 = fully_connected(x, 512, activation_fn=tf.nn.relu, scope="fc1")
            fc2 = fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope="fc2")
            fc3 = fully_connected(fc2, 128, activation_fn=tf.nn.relu, scope="fc3")
            fc4 = fully_connected(fc3, 64, activation_fn=tf.nn.relu, scope="fc4")
            fc5 = fully_connected(fc4, 1, activation_fn=None, scope="fc5")
        return fc5

    def vars(self):

        return [var for var in tf.global_variables() if self.name in var.name]