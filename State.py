
import numpy as np

class InitialDiscreteState(object):
    def __init__(self,states,probs):
        self.states=states
        self.probs=probs

    def generate(self):
        idx = np.random.choice(len(self.states),p=self.probs)
        return self.states[idx]