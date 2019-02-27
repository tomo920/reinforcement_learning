import numpy as np
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
        #initialize policy network
        self.inputs = Input(shape=(state_size, ))
        o1 = Dense(hidden1, activation='relu')(self.inputs)
        o2 = Dense(hidden2, activation='relu')(o1)
        o = Dense(action_size, activation='tanh')(o2)
        mu = Lambda(lambda x: (action_high-action_low)*x/2+(action_high+action_low)/2, output_shape=(action_size, ))(o)
        self.mu = Model(inputs=self.inputs, outputs=mu)
        #loss function
        self.weights = tf.placeholder(tf.float32, [None, 1])
        self.action = tf.placeholder(tf.int64, [None, action_size])
        log_pi = -(tf.reduce_sum(((self.action-self.mu.outputs)/np.exp(log_sigma))**2+2*log_sigma, 1)+action_size*np.log(2*np.pi))/2
        loss = -tf.matmul(log_pi, self.weights)/sumple_num
        #optimize
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def choose_action(self, state):
        state = np.array([state])
        return self.mu.predict(state)[0]+np.exp(log_sigma)*np.random.randn(self.action_size)

    def train(self, state_batch, action_batch, weight_bach):
        self.sess.run(self.optimize, feed_dict={self.inputs:state_batch, self.action:action_batch, self.weights:weight_bach})
