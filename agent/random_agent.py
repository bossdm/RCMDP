import numpy as np
from collections import namedtuple
import os
import sys, os
RCMDP_path=os.environ["PROJECT_DIR"]+'/RCMDP/'
sys.path.extend([RCMDP_path])
from Utils import check_folder

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

""" source: https://github.com/aidudezzz/deepworlds/blob/dev/examples/cartpole/cartpole_discrete/controllers/robot_supervisor_manager/agent/PPO_agent.py"""
class RandomAgent:
    """
    PPOAgent implements the PPO RL algorithm (https://arxiv.org/abs/1707.06347).
    It works with a set of discrete actions.
    It uses the Actor and Critic neural network classes defined below.
    """

    def __init__(self, numberOfActorOutputs,uncertainty_set):
        self.outputs=numberOfActorOutputs
        self.uncertainty_set=uncertainty_set
        self.buffer=[]

    def work(self, agentInput, test=False):
        """
        random agents simply selects a random action
        """
        return np.random.choice(list(range(self.outputs))), None, 1.0/self.outputs

    def save(self, path,episode):
        """
        """
        folder=path+"/episode"+str(episode)
        check_folder(folder)
        self.uncertainty_set.save(folder)

    def load(self, path,episode):
        """
        """
        folder = path + "/episode" + str(episode)
        self.uncertainty_set.load(folder)

    def storeTransition(self, transition):
        """
        Stores a transition in the buffer to be used later.

        :param transition: contains state, action, action_prob, reward, next_state
        :type transition: namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
        """
        self.buffer.append(transition)

    def trainStep(self, batchSize=None):
        """
        Performs a training step for the actor and critic models, based on transitions gathered in the
        buffer. It then resets the buffer.

        :param batchSize: Overrides agent set batch size, defaults to None
        :type batchSize: int, optional
        """
        del self.buffer[:]

    def random_state(self,s_index,a):
        return None, None, None


