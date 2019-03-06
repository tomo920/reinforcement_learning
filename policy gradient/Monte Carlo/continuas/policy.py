import numpy as np
import math
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Lambda, noise
import keras.backend as K

log_sigma = -0.5
hidden1 = 64
hidden2 = 64
learning_rate = 1e-4

class Policy:
    def __init__(self, sess, state_size, action_size, action_high, action_low, sample_num):
        self.sess = sess
        self.action_size = action_size
        self.sample_num = sample_num
        self.weights = tf.placeholder(tf.float32, [None, 1])
        self.action = tf.placeholder(tf.float32, [None, action_size])
        #initialize policy network
        self.state = Input(shape=(state_size, ))
        self.action = Input(shape=(action_size, ))
        self.weights = Input(shape=(1, ))
        o1 = Dense(hidden1, activation='relu')(self.state)
        o2 = Dense(hidden2, activation='relu')(o1)
        o = Dense(action_size, activation='tanh')(o2)
        mu = Lambda(lambda x: (action_high-action_low)*x/2+(action_high+action_low)/2, output_shape=(action_size, ))(o)
        outputs = Lambda(lambda x: x+np.exp(log_sigma)*np.random.randn(self.action_size), output_shape=(action_size, ))(mu)
        loss = Lambda(self.loss, output_shape=(1, ))([mu, self.action, self.weights])
        self.actor = Model(inputs=self.state, outputs=outputs)
        #optimize
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def loss(self, args):
        mu, action, weights = args
        log_pi = -(tf.reduce_sum(tf.square((action-mu)/tf.exp(log_sigma))+2*log_sigma, axis=1)+self.action_size*tf.log(2*np.pi))/2
        return -np.dot(log_pi, weights)/self.sample_num

    def choose_action(self, state):
        state = np.array([state])
        return self.actor.predict(state)[0]

    def train(self, state_batch, action_batch, weight_bach):
        self.sess.run(self.optimize, feed_dict={self.state:state_batch, self.action:action_batch, self.weights:weight_bach})
