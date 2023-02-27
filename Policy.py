

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.models import load_model
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

    def select_action(self,s, deterministic):

            with tf.GradientTape() as tape:
                probs = self.pi(np.array([s]))[0]
                grad = tape.gradient(probs, self.pi.trainable_weights)
            probs = K.eval(probs)
            if deterministic:
                return np.argmax(probs).item(), None, 1.0
            else:
                a = np.random.choice(len(probs), p=probs)
                pr = probs[a]
                grad = [g / pr for g in
                        grad]  # (and divide by the probability of the action to compensate for more frequent updates)
                return a, grad, probs

    def save(self,savefile):
        print("saving policy at ",savefile)
        self.pi.save(savefile)

    def load(self,loadfile):
        print("loading policy from ", loadfile)
        self.pi = load_model(loadfile)



