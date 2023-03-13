

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.models import load_model
import numpy as np

class Critic(object):

    def __init__(self,D_S):
        self.C = Sequential()
        self.C.add(Dense(20, name="hidden1",activation="relu",kernel_initializer='uniform', input_shape=(D_S,)))
        self.C.add(Dense(1, name="output",activation="linear"))
        self.C.compile(optimizer="rmsprop",loss="MSE")
        self.batch_x = []
        self.batch_y = []

    def params(self):
        return self.C.trainable_weights
    def summary(self):
        return self.C.summary()

    def add_to_batch(self,x,y):
        self.batch_x.append(x)
        self.batch_y.append(y)
    def train(self):
        self.C.train_on_batch(np.array(self.batch_x),np.array(self.batch_y))
        self.batch_x=[]
        self.batch_y=[]
    def predict(self,s):

        value = self.C.predict(s,batch_size=1)
        return value

    def save(self,savefile):
        print("saving critic at ",savefile)
        self.C.save(savefile)

    def load(self,loadfile):
        print("loading critic from ", loadfile)
        self.C = load_model(loadfile)



