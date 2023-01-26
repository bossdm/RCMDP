""" implements a scalable version of Robust Constrained Policy Gradient (Russel et al.)
    where scalability is due to avoiding the inner problem in favour of a perturbation scheme

 """

from UncertaintySet import *
from CMDP import *

class RCPG(object):
    def __init__(self,pi, real_CMDP, uncertainty_set,optimiser_theta,optimiser_lbda,
                 sim_iterations,real_iterations,train_iterations,lr1,lr2,logfile,simlogfile):
        """
        :param pi:  policy, a keras model
        :param real_cmdp: the real CMDP
        :param uncertainty_set: the chosen uncertainty_set
        :param optimiser_theta: the chosen optimiser for the policy
        :param optimiser_lbda: the chosen optimiser for the lagrangian multiplier
        :param sim_iterations: the number of iterations for offline optimisation during the training
        :param real_iterations: the number of iterations of real trajectories during the training
        :param train_iterations: the number of training iterations
        :param lr1: learning schedule for L
        :param lr2: learning schedule for lambda
        :param logfile for optimisation
        :param logfile for simulation
        """
        self.pi = pi
        self.real_CMDP = real_CMDP
        self.uncertainty_set = uncertainty_set
        self.optimiser_theta = optimiser_theta
        self.optimiser_lbda = optimiser_lbda
        self.sim_iterations = sim_iterations
        self.real_iterations = real_iterations
        self.train_iterations = train_iterations
        self.lbda = tf.Variable(0.1)
        self.lr1=lr1
        self.lr2=lr2
        self.n = 0# count the number of iterations
        self.logfile=logfile
        self.logfile.write("L \t lambda \n")
        self.logfile.flush()
        self.simlogfile=simlogfile

    def train(self):
        for t in range(self.train_iterations):
            if self.real_iterations > 0: # narrow the uncertainty set while you are going over real samples
                self.real_samples()
            self.offline_optimisation()

    def test(self):
       self.real_CMDP.episode(self.pi)

    def real_samples(self):
        for it in range(self.real_iterations):
            trajectory,t = self.real_CMDP.episode(self.pi)
            self.uncertainty_set.add_visits(trajectory)
        self.uncertainty_set.set_params()
    def offline_optimisation(self):
        self.sim_CMDP = RobustCMDP.from_CMDP(self.real_CMDP,self.simlogfile,self.uncertainty_set)
        for it in range(self.sim_iterations):
            trajectory,t = self.sim_CMDP.episode(self.pi)
            T_stop = t - 1
            V = 0
            C = 0
            self.n += 1 # count
            for t in range(T_stop,-1,-1):
                s,a,r,c,s_next,grad, grad_adv, probs_adv = trajectory[t]
                V = r + self.sim_CMDP.gamma * V
                C = c + self.sim_CMDP.gamma * C
                actual_lbda = tf.clip_by_value(tf.exp(self.lbda), clip_value_min=0, clip_value_max=10000)
                eta1=self.lr1(self.n)
                eta2=self.lr2(self.n)
                L = -(V-actual_lbda*C)
                update = [eta1 * L * g for g in grad]   # dL/d\pi * d\pi/d\theta
                self.update_theta(update)
                update_l = -eta2*(C - self.sim_CMDP.d)           # dL/d\lambda
                self.update_lbda(update_l)
                if self.sim_CMDP.uncertainty_set.adversarial: # min L s.t. ||P-P*|| <= alpha
                    # try to make the agent fail the objective
                    L_adv,lbda_adv = self.sim_CMDP.uncertainty_set.update_adversary(eta1,eta2,s,a,L,grad_adv,probs_adv)
                    self.logfile.write("%.4f \t %.4f \t %.4f \t %.4f \n" % (K.eval(L), K.eval(actual_lbda),L_adv,lbda_adv))
                    self.logfile.flush()
                else:
                    self.logfile.write("%.4f \t %.4f \n" % (K.eval(L), K.eval(actual_lbda)))
                    self.logfile.flush()
                # print("L",L)
                # print("lbda",actual_lbda)

        print("offline optimisation ended")
        self.pi.summary()

    def update_theta(self,grad_theta):
        self.optimiser_theta.apply_gradients(zip(grad_theta, self.pi.params()))  # increase iteration by one

    def update_lbda(self,grad_lbda):
        self.optimiser_lbda.apply_gradients([(grad_lbda, self.lbda)])  # increase iteration by one




