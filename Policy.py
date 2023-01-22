

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
import numpy as np

class StochasticPol(object):

    def __init__(self,D_S,D_A):
        self.pi = Sequential()
        self.pi.add(Dense(20, name="hidden1",activation="relu",kernel_initializer='uniform', input_shape=(D_S,)))
        self.pi.add(Dense(D_A, name="output",activation="softmax"))

    def params(self):
        return self.pi.trainable_weights
    def summary(self):
        return self.pi.summary()

    def select_action(self,s):
        with tf.GradientTape() as tape:
            probs = self.pi(np.array([s]))[0]
            grad = tape.gradient(probs, self.pi.trainable_weights)
        probs = K.eval(probs)
        a = np.random.choice(len(probs), p=probs)
        pr = probs[a]
        grad = [ g /pr for g in grad] # (and divide by the probability of the action to compensate for more frequent updates)
        return a, grad, probs

