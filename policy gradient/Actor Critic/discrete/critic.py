import numpy as np
import tensorflow as tf
from keras.models import Model, clone_model
from keras.layers import Input, Dense, Lambda
import keras.backend as K

hidden1 = 64
hidden2 = 64
learning_rate = 1e-4
gamma = 0.98
use_targetnetwork = True

class Critic:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        #initialize critic network
        state = Input(shape=(state_size, ))
        action = Input(shape=(1, ), dtype='int32')
        o1 = Dense(hidden1, activation='relu')(state)
        o2 = Dense(hidden2, activation='relu')(o1)
        o = Dense(action_size, activation='linear')(o2)
        q = Lambda(self.get_q, output_shape=(1, ))([o, action])
        self.critic = Model(inputs=[state, action], outputs=q)
        #initialize target network
        self.target = clone_model(self.critic)
        self.target.set_weights(self.critic.get_weights())
        #compile critic network
        self.critic.compile(optimizer='adam', loss='mse')

    def get_q(self, args):
        q_sa, action = args
        action = tf.reshape(action, [-1])
        mask = tf.one_hot(action, depth=self.action_size, dtype=tf.float32)
        return tf.reduce_sum(q_sa*mask, axis=-1)

    def q_value(self, state_batch, action_batch):
        return self.critic.predict([state_batch, action_batch])

    def next_q_value(self, next_state_batch, next_action_batch):
        if use_targetnetwork:
            return self.target.predict([next_state_batch, next_action_batch])
        else:
            return self.critic.predict([next_state_batch, next_action_batch])

    def train(self, state_batch, action_batch, next_state_batch, next_action_batch, reward_batch, done_batch):
        q_target = reward_batch+gamma*(1-done_batch)*self.next_q_value(next_state_batch, next_action_batch)
        self.critic.train_on_batch([state_batch, action_batch], q_target)
