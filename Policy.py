
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.models import load_model
import numpy as np

class StochasticPol(object):

    def __init__(self,D_S,D_A):
        self.pi = Sequential()
        self.pi.add(Dense(100, name="hidden1",activation="relu",kernel_initializer='uniform', input_shape=(D_S,)))
        self.pi.add(Dense(D_A, name="output",activation="softmax"))
    def params(self):
        return self.pi.trainable_weights
    def summary(self):
        return self.pi.summary()
    def predict(self,s):
        return self.pi(s)

    def select_action(self,s, deterministic=False,nominal=None):
        if deterministic:  # testing
            probs = K.eval(self.pi(np.array([s]))[0])
            a = np.argmax(probs).item()
            # print("s in select", s)
            # print("a in select", a)
            return a, None, 1.0
        else:
            with tf.GradientTape(persistent=True) as tape:
                    probs = self.pi(np.array([s]))
                    log_probs = tf.math.log(probs+1e-8)
                    a = tf.random.categorical(log_probs, 1)[0,0]
                    logprob = log_probs[0,a]
                    entropy = -tf.reduce_sum(probs[0]*log_probs[0])  # just to make sure it scales well use lambda here as well
                    #print("entropy ", K.eval(entropy))
                    #print(probs,a)
                    if nominal is not None:
                        delta_P = tf.norm(probs[0] - nominal, ord=1)  # L1 norm
            grad = tape.gradient(logprob, self.pi.trainable_weights) # grad log pi(a|s) = grad pi(a|s)/ pi(a|s)
            grad_H = tape.gradient(entropy, self.pi.trainable_weights) # grad log pi(a|s) = grad pi(a|s)/ pi(a|s)
            if nominal is not None:
                #grad_delta_P = tape.gradient(delta_P, self.pi.trainable_weights) #
                grad =(grad, grad_H,delta_P)
            else:
                grad = (grad,grad_H)
            del tape
            # print("s in select", s)
            # print("a in select", K.eval(a))
            return K.eval(a), grad, probs

    def save(self,savefile):
        print("saving policy at ",savefile)
        self.pi.save(savefile)

    def load(self,loadfile):
        print("loading policy from ", loadfile)
        self.pi = load_model(loadfile)



