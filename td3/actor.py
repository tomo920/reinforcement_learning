import numpy as np
import tensorflow as tf
from keras.models import Model, clone_model
from keras.layers import Input, Dense, Lambda
import keras.backend as K

hidden1 = 400
hidden2 = 300
learning_rate = 1e-4
tau = 1e-3

class Actor:
    def __init__(self, sess, state_size, action_size, action_high, action_low):
        self.sess = sess
        optimizer = tf.train.AdamOptimizer(learning_rate)
        K.set_session(sess)
        #initialize evaluation network
        self.inputs = Input(shape=(state_size, ))
        o1 = Dense(hidden1, activation='relu')(self.inputs)
        o2 = Dense(hidden2, activation='relu')(o1)
        o = Dense(action_size, activation='tanh')(o2)
        scaled = Lambda(lambda x: (action_high-action_low)*x/2+(action_high+action_low)/2, output_shape=(action_size, ))(o)    #scaleout
        self.eval_model = Model(inputs=self.inputs, outputs=scaled)
        eval_params = self.eval_model.trainable_weights
        #initialize target network
        self.target_model = clone_model(self.eval_model)
        self.target_model.set_weights(self.eval_model.get_weights())
        self.action_grad_batch = tf.placeholder(tf.float32, [None, action_size])
        grads = tf.gradients(self.eval_model.outputs, eval_params, self.action_grad_batch)
        self.optimize = optimizer.apply_gradients(zip(grads, eval_params))

    def action(self, state_batch):
        return self.eval_model.predict(state_batch)

    def next_action(self, next_state_batch):
        return self.target_model.predict(next_state_batch)

    def train(self, state_batch, action_grad_batch):
        self.sess.run(self.optimize, feed_dict={self.inputs:state_batch, self.action_grad_batch:action_grad_batch})

    def update_target(self):
        eval_params = self.eval_model.get_weights()
        target_params = self.target_model.get_weights()
        for i in range(len(target_params)):
            target_params[i] = tau*eval_params[i]+(1-tau)*target_params[i]
        self.target_model.set_weights(target_params)
