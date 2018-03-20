import tensorflow as tf
from tensorflow.contrib.layers import *

class Generator(object):
    def __init__(self):
        self.name = 'generator'
        # self.output_dim = output_dim

    def __call__(self, z):
        with tf.variable_scope(self.name):
            fc1 = fully_connected(z, 128, activation_fn=tf.nn.relu, scope="fc1")
            fc2 = fully_connected(fc1, 256, activation_fn=tf.nn.relu, scope="fc2")
            fc3 = fully_connected(fc2, 512, activation_fn=tf.nn.relu, scope="fc3")
            fc4 = fully_connected(fc3, 28*28, activation_fn=tf.sigmoid, scope="fc4")
        return fc4
        #     h0 = fully_connected(z, 4*4*1024, activation_fn=tf.nn.relu, scope='h0')
        #     rshape0 = tf.reshape(h0, [-1, 4, 4, 1024])
        #     conv1 = tf.layers.conv2d_transpose(rshape0, 512, kernel_size=2,
        #                 strides=2, kernel_initializer=tf.initializers.random_normal)
        #     # print('conv1 shape:'+conv1.shape())
        #     print(conv1)
        #     conv2 = tf.layers.conv2d_transpose(conv1, 1, kernel_size=14,
        #                 strides=2, kernel_initializer=tf.initializers.random_normal)
        #     # print('conv2 shape:'+conv2.shape())
        #     print(conv2)
        #     # conv3 = tf.layers.conv2d_transpose(conv2, 128, kernel_size=2,
        #     #             strides=2, kernel_initializer=tf.initializers.random_normal)
        #     # # print('conv3 shape:'+conv3.shape())
        #     # print(conv3)
        #     # conv4 = tf.layers.conv2d_transpose(conv3, 1, kernel_size=2,
        #     #             strides=2, kernel_initializer=tf.initializers.random_normal)
        #     conv4 = tf.reshape(conv2, [-1, 28*28])
        # return conv4

    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]