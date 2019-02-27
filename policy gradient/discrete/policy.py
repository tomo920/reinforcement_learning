import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
import keras.backend as K

hidden1 = 64
hidden2 = 64
learning_rate = 1e-4

class Policy:
    def __init__(self, sess, state_size, action_size, sample_num):
        self.sess = sess
        #initialize policy network
        self.inputs = Input(shape=(state_size, ))
        o1 = Dense(hidden1, activation='relu')(self.inputs)
        o2 = Dense(hidden2, activation='relu')(o1)
        outputs = Dense(action_size, activation='linear')(o2)
        self.model = Model(inputs=self.inputs, outputs=outputs)
        #loss function
        self.weights = tf.placeholder(tf.float32, [None, 1])
        self.action = tf.placeholder(tf.int64, [None])
        self.mask = tf.one_hot(self.action, depth=action_size, dtype=tf.float32)
        loss = -tf.reduce_sum(self.mask*(self.model.outputs-tf.log(tf.reduce_sum(tf.exp(self.model.outputs), axis=1)))*self.weights)/sample_num
        #optimize
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def choose_action(self, state):
        state = np.array([state])
        return K.get_value(tf.random.multinomial(self.model.predict(state), 1))[0][0]

    def train(self, state_batch, action_list, weight_bach):
        self.sess.run(self.optimize, feed_dict={self.inputs:state_batch, self.action:action_list, self.weights:weight_bach})
