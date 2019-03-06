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
        #initialize two Q network
        self.state =  Input(shape=(state_size, ))
        self.action = Input(shape=(action_size, ))
        inputs = Concatenate()([self.state, self.action])
        o1 = Dense(hidden1, activation='relu')(inputs)
        o2 = Dense(hidden2, activation='relu')(o1)
        o = Dense(1, activation='linear')(o2)
        self.q1 = Model(inputs=[self.state, self.action], outputs=o)
        self.q2 = clone_model(self.q1)
        #initialize two target network
        self.target1 = clone_model(self.q1)
        self.target1.set_weights(self.q1.get_weights())
        self.target2 = clone_model(self.q2)
        self.target2.set_weights(self.q2.get_weights())
        #compile two Q network
        self.q1.compile(optimizer='adam', loss='mse')
        self.q2.compile(optimizer='adam', loss='mse')
        loss = -tf.reduce_mean(self.q1.outputs)
        self.grad = K.gradients(loss, self.action)

    def q_value(self, q, state_batch, action_batch):
        return q.predict([state_batch, action_batch])

    def next_q_value(self, q, next_state_batch, next_action_batch):
        return q.predict([next_state_batch, next_action_batch])

    def action_grad(self, state_batch, action_batch):
        return self.sess.run(self.grad, feed_dict={self.state:state_batch, self.action:action_batch})[0]

    def train(self, state_batch, action_batch, q_target):
        self.q1.train_on_batch([state_batch, action_batch], q_target)
        self.q2.train_on_batch([state_batch, action_batch], q_target)

    def update_target(self):
        for q, target in zip([self.q1, self.q2], [self.target1, self.target2]):
            q_params = q.get_weights()
            target_params = target.get_weights()
            for i in range(len(target_params)):
                target_params[i] = tau*q_params[i]+(1-tau)*target_params[i]
            target.set_weights(target_params)
