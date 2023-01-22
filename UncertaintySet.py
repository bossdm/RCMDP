
import scipy
import numpy as np
from random import random

class DummyUncertaintySet(object):
    def __init__(self,states,actions):
        self.S=len(states)
        self.A=len(actions)
        self.data = np.zeros((self.S,self.A,self.S))  # SxAxS frequency table
        self.states=states
        self.actions=actions
        self.nominal=np.zeros((self.S,self.A,self.S)) + 1/self.S # uniform at the start
    def add_visits(self,trajectory):
        for s_index, a_index, _r, _c, s_next, _grad in trajectory:
            s_next_index = self.states.index(s_next)
            self.data[s_index, a_index, s_next_index] += 1  # add trajectory to the data counts
    def set_params(self):
        self.visits = np.sum(self.data,
                             axis=2)  # sum over third axis (don't care about next state, only the sa-visitations
        for s_index in range(self.S):
            for a_index in range(self.A):
                if self.visits[s_index, a_index] > 0: #otherwise keep at uniform random
                    self.nominal[s_index, a_index] = self.data[s_index, a_index] / self.visits[s_index, a_index]
    def random_state(self,s,a):
        try:
            s_next_index = np.random.choice(self.S,p=self.nominal[s,a])
        except Exception as e:
            print(self.nominal[s,a])
            print(e)
        s_next=self.states[s_next_index]
        return s_next

class HoeffdingSet(object):
    """
    uncertainty set based on Hoeffding
    """
    def __init__(self,delta, states,actions):
        self.delta = delta # desired confidence level
        self.S=len(states)
        self.A=len(actions)
        self.data = np.zeros((self.S,self.A,self.S))  # SxAxS frequency table
        self.states=states
        self.actions=actions
        self.nominal=np.zeros((self.S,self.A,self.S)) + 1/self.S # uniform at the start
        self.alpha=np.zeros((self.S,self.A))
    def add_visits(self,trajectory):
        for s_index,a_index,_r,_c,s_next,_grad in trajectory:
            s_next_index = self.states.index(s_next)
            self.data[s_index,a_index,s_next_index] += 1   # add trajectory to the data counts
    def set_params(self):
        """

        :param
        :return:
        """

        self.visits = np.sum(self.data,axis=2) # sum over third axis (don't care about next state, only the sa-visitations
        for s_index in range(self.S):
            for a_index in range(self.A):
                if self.visits[s_index, a_index] > 0: #otherwise keep at uniform random
                    self.nominal[s_index, a_index] = self.data[s_index, a_index] / self.visits[s_index, a_index]
                    self.alpha[s_index,a_index] = self.compute_alpha(s_index,a_index)
                    print(r"Hoeffding uncertainty set for ",  (s_index,a_index), r"has $\bar{\pi}$=", self.nominal[s_index,a_index], r" and $\alpha$=",self.alpha[s_index,a_index])
        self.min = self.nominal - self.alpha
        self.range = 2 * self.alpha



    def random_probs(self,s,a):
        probs=[None for i in range(self.S)]
        for i in range(self.S):
            probs[i] = self.min[s,a] + np.random.random(self.S)*self.range[s,a]

        return probs

    def random_state(self,s,a):
        probs=self.random_probs(s,a)
        s_next_index = np.random.choice(self.S,p=probs)
        s_next=self.states[s_next_index]
        return s_next

    def compute_alpha(self,s,a):
        """
        :return:
        """
        return np.sqrt(2/self.visits[s,a]*np.log(self.S*self.A*2**self.S/self.delta))




class BCIUncertaintySet(object):   # see algorithm 2 in appendix of BCI paper
    """
    given data set D, BCI solves the optimisation problem for tight ambiguity sets:

    min {phi: P(|| \bar{P}_s,a - P^*_s,a || > phi | D ) < delta / SA
    """
    def __init__(self,posterior,delta,n):
        self.posterior=posterior
        self.delta=delta
        self.n = n
        self.nominal = np.zeros((self.S,self.A,self.S)) + 1/self.S # uniform at the start
        self.alpha = np.zeros((self.S, self.A))

    def add_visits(self, trajectory):
        self.posterior.update(trajectory)
    def set_params(self):
        for s_index in range(self.S):
            for a_index in range(self.S):
                probslist = []
                for i in range(self.n):
                    probs = self.posterior.generate(s_index,a_index)
                    probslist.append(probs)
                self.nominal[s_index,a_index] = np.mean(probslist)
                dists=np.sum(np.abs(np.array(probslist) - self.nominal[s_index,a_index]),axis=1)
                sorted_dists = np.sort(dists)[::-1] # sorted in increasing order
                idx = self.delta*self.n
                self.alpha[s_index,a_index] = sorted_dists[idx]



