import tensorflow as tf
from keras.models import Model, clone_model
from keras.layers import Input, Dense, Concatenate
import keras.backend as K

hidden1 = 400
hidden2 = 300
learning_rate = 1e-3
tau = 1e-3

class Critic:
    def __init__(self, sess, state_size, action_size):
        self.sess = sess
        K.set_session(sess)
        #initialize evaluation network
        self.state =  Input(shape=(state_size, ))
        self.action = Input(shape=(action_size, ))
        inputs = Concatenate()([self.state, self.action])
        o1 = Dense(hidden1, activation='relu')(inputs)
        o2 = Dense(hidden2, activation='relu')(o1)
        o = Dense(1, activation='linear')(o2)
        self.eval_model = Model(inputs=[self.state, self.action], outputs=o)
        self.eval_params = self.eval_model.trainable_weights
        #initialize target network
        self.target_model = clone_model(self.eval_model)
        self.target_model.set_weights(self.eval_model.get_weights())
        #compile evaluation network
        self.eval_model.compile(optimizer='adam', loss='mse')
        loss = -tf.reduce_mean(self.eval_model.outputs)
        self.grad = K.gradients(loss, self.action)

    def q_value(self, state_batch, action_batch):
        return self.eval_model.predict([state_batch, action_batch])

    def next_q_value(self, next_state_batch, next_action_batch):
        return self.target_model.predict([next_state_batch, next_action_batch])

    def action_grad(self, state_batch, action_batch):
        return self.sess.run(self.grad, feed_dict={self.state:state_batch, self.action:action_batch})[0]

    def train(self, state_batch, action_batch, q_target):
        self.eval_model.train_on_batch([state_batch, action_batch], q_target)

    def update_target(self):
        eval_params = self.eval_model.get_weights()
        target_params = self.target_model.get_weights()
        for i in range(len(target_params)):
            target_params[i] = tau*eval_params[i]+(1-tau)*target_params[i]
        self.target_model.set_weights(target_params)
