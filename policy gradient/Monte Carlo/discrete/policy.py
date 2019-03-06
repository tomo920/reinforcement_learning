import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Lambda
import keras.backend as K

hidden1 = 64
hidden2 = 64
learning_rate = 1e-4

class Policy:
    def __init__(self, sess, state_size, action_size, sample_num):
        self.sess = sess
        self.action_size = action_size
        #initialize policy network
        state = Input(shape=(state_size, ))
        weights = Input(shape=(1, ))
        action = Input(shape=(1, ), dtype='int32')
        o1 = Dense(hidden1, activation='relu')(state)
        o2 = Dense(hidden2, activation='relu')(o1)
        o = Dense(action_size, activation='linear')(o2)
        outputs = Lambda(lambda x: tf.random.multinomial(x, 1), output_shape=(1, ))(o)
        self.actor = Model(inputs=state, outputs=outputs)
        logpi_w = Lambda(self.get_loss, output_shape=(1, ))([o, action, weights])
        self.trainer = Model(inputs=[state, action, weights], outputs=logpi_w)
        self.trainer.compile(optimizer='adam', loss=self.loss)

    def get_loss(self, args):
        logits, action, weights = args
        action = tf.reshape(action, [-1])
        mask = tf.one_hot(action, depth=self.action_size, dtype=tf.float32)
        logpi = tf.reduce_sum((logits-tf.transpose([K.logsumexp(logits, axis=-1)]))*mask, axis=-1)
        return tf.transpose([logpi])*weights

    def loss(self, y_true, y_predict):
        return -K.mean(y_predict, axis=-1)

    def choose_action(self, state):
        state = np.array([state])
        return self.actor.predict(state)[0][0]

    def train(self, state_batch, action_batch, weight_bach):
        self.trainer.train_on_batch([state_batch, action_batch, weight_bach], action_batch)
