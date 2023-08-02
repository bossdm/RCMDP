from RCMDP.RCPG import *

class RCPG_Agent(RCPG):
    """
    RCPG Agent implements the RCPG RL algorithm (https://github.com/bossdm/RCMDP.git).
    It works with a set of discrete actions.
    It uses the Actor and Critic neural network classes defined below.
    """

    def __init__(self, no_noise,pi, real_cmdp, uncertainty_set, optimiser_theta, optimiser_lbda,
                 sim_iterations, real_iterations, train_iterations, lr1, lr2, logfile, simlogfile):
        RCPG.__init__(self,pi, real_cmdp, uncertainty_set, optimiser_theta, optimiser_lbda,
                 sim_iterations, real_iterations, train_iterations, lr1, lr2, logfile, simlogfile)
        self.buffer = []
        self.no_noise = no_noise
        self.trajectories=[]

    @classmethod
    def from_RCPG(cls,rcpg,no_noise):
        agent = RCPG_Agent(no_noise,rcpg.pi, rcpg.real_CMDP,rcpg.uncertainty_set, rcpg.optimiser_theta, rcpg.optimiser_lbda,
                 rcpg.sim_iterations, rcpg.real_iterations, rcpg.train_iterations, rcpg.lr1, rcpg.lr2, rcpg.logfile, rcpg.simlogfile)
        return agent

    def work(self, s, test,random=False):
        """
        Forward pass. Depending on the type_ argument, it either explores by sampling its actor's
        softmax output, or eliminates exploring by selecting the action with the maximum probability (argmax).

        :param agentInput: The actor neural network input vector
        :type agentInput: vector
        :param type_: "selectAction" or "selectActionMax", defaults to "selectAction"
        :type type_: str, optional
        """
        if random:
            a_index = np.random.choice(list(range(len(self.real_CMDP.actions))))
            grad = None
            probs = 1.0/len(self.real_CMDP.actions)
            return a_index, grad, probs
        else:
            return self.pi.select_action(s, deterministic=test)

    def save(self, path, episode):
        """
        Save models in the path provided.

        :param path: path to save the models
        :type path: str
        """
        RCPG.save(self,path+"/episode"+str(episode)+"/stored")

    def load(self, path, episode):
        """
        Load models from the path provided.

        :param path: path where the models are saved
        :type path: str
        """
        RCPG.load(self,path+"/episode"+str(episode)+"/stored")

    def load_from_path(self, path):
        """
        Load models from the path provided.

        :param path: path where the models are saved
        :type path: str
        """
        RCPG.load(self,path)

    def storeTransition(self, transition):
        """
        Stores a transition in the buffer to be used later.

        :param transition: contains state, action, action_prob, reward, next_state
        :type transition: namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
        """
        self.buffer.append(transition)


    def trainStep(self, batchSize=None):
        """
        Performs a training step based on transitions gathered in the
        buffer. It then resets the buffer.

        :param batchSize: Overrides agent set batch size, defaults to None
        :type batchSize: int, optional
        """
        # Default behaviour waits for buffer to collect at least one batch_size of transitions
        self.update(self.buffer, self.gamma, self.d)
        del self.buffer[:]

    def testStep(self):
        """
        """
        self.trajectories.append(self.buffer)
        self.buffer=[]

    def random_state(self,s_index,a):
        """
        Returns the worst-case state given state-action pair (s,a)
        """
        if self.no_noise:
            return None, None, None
        else:
            return self.uncertainty_set.random_state(s_index,a)