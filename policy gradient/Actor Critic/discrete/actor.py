import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Lambda
import keras.backend as K

hidden1 = 64
hidden2 = 64
learning_rate = 1e-4

class Actor:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        #initialize actor network
        state = Input(shape=(state_size, ))
        weights = Input(shape=(1, ))
        action = Input(shape=(1, ), dtype='int32')
        o1 = Dense(hidden1, activation='relu')(state)
        o2 = Dense(hidden2, activation='relu')(o1)
        o = Dense(action_size, activation='linear')(o2)
        outputs = Lambda(lambda x: tf.random.multinomial(x, 1), output_shape=(1, ))(o)
        self.actor = Model(inputs=state, outputs=outputs)
        #make loss
        logpi_w = Lambda(self.get_loss, output_shape=(1, ))([o, action, weights])
        loss = -K.mean(logpi_w)
        update = K.optimizers.Adam.get_updates(loss=loss, params=self.actor.trainable_weights)
        self.train = K.function(inputs=[state, action, weights], outputs=loss, updates=update)

    def get_loss(self, args):
        logits, action, weights = args
        action = tf.reshape(action, [-1])
        mask = tf.one_hot(action, depth=self.action_size, dtype=tf.float32)
        logpi = tf.reduce_sum((logits-tf.transpose([K.logsumexp(logits, axis=-1)]))*mask, axis=-1)
        logpi_w = tf.transpose([logpi])*weights
        return logpi_w

    def action(self, state):
        return self.actor.predict(state)

    def train(self, state_batch, action_batch, weight_bach):
        self.train([state_batch, action_batch, weight_bach])
