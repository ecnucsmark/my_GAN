import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from generator import Generator
from discriminator import Discriminator
import time
import numpy as np

from scipy.misc import imsave
from visualize import *


tf.flags.DEFINE_integer('batch_size', 64, 'size of batch')

FLAGS = tf.flags.FLAGS

class DCGAN(object):
    def __init__(self, input_width, input_height):
        self.input_img = tf.placeholder(tf.float32, [FLAGS.batch_size,
                                        input_width*input_height], name='input_image')
        self.z = tf.placeholder(tf.float32, [FLAGS.batch_size, 100], name='z')

        # gen = Generator(output_dim=input_width*input_height)
        gen = Generator()
        dis = Discriminator()

        self.x_noise = gen(self.z)
        d_noise = dis(self.x_noise)
        d_real = dis(self.input_img)
        epsilon = tf.random_normal([], 0.0, 1.0)
        x_hat = epsilon*self.input_img + (1. - epsilon)*self.x_noise

        self.dis_loss = tf.reduce_mean(d_real) - tf.reduce_mean(d_noise)

        penalty = tf.gradients(dis(x_hat), x_hat)
        penalty = tf.sqrt(tf.reduce_sum(tf.square(penalty), axis=1))
        penalty = tf.reduce_mean(tf.square(penalty - 1.0) * 10)

        self.dis_loss += penalty

        self.gen_loss = tf.reduce_mean(d_noise)

        print(dis.vars)
        print(gen.vars)

        with tf.variable_scope('train_op'):
            self.dis_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9).minimize(self.dis_loss, var_list=dis.vars())
            self.gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9).minimize(self.gen_loss, var_list=gen.vars())



if __name__ == '__main__':
    start_time = time.time()
    mnist = input_data.read_data_sets('../datasets/MNIST/', one_hot=True)

    input_height = 28
    input_width = 28

    model = DCGAN(input_width, input_height)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img_index = 0

        for step in range(100000):
            d_loss = 0.0
            for _ in range(5):
                batch = mnist.train.next_batch(FLAGS.batch_size)
                _, d_loss = sess.run(fetches=[model.dis_train_op, model.dis_loss],
                                     feed_dict={model.input_img:batch[0],
                                    model.z:np.random.uniform(-1., 1., [FLAGS.batch_size, 100])})

            _, g_loss = sess.run(fetches=[model.gen_train_op, model.gen_loss],
                                 feed_dict={model.input_img:batch[0],
                                    model.z:np.random.uniform(-1., 1., [FLAGS.batch_size, 100])})


            if step % 1000 == 0:
                print("step {}, d_loss {}, g_loss {}".format(step, d_loss, g_loss))
                img = sess.run(model.x_noise, {model.z: np.random.uniform(-1., 1., [FLAGS.batch_size, 100])})
                img = np.reshape(img, [-1, 28, 28, 1])
                img = grid_transform(img, [28, 28, 1])
                imsave('imgs/%08d.png' % img_index, img)
                img_index += 1

    end_time = time.time()
    print("end with {} hours".format((end_time - start_time) / 3600.0))