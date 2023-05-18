""" implements a scalable version of Robust Constrained Policy Gradient (Russel et al.)
    where scalability is due to avoiding the inner problem in favour of a perturbation scheme

 """

from RCMDP.UncertaintySet import *
from RCMDP.CMDP import *
import pickle
import time
import keras.backend as K
import numpy as np
PRINTING=False
K.set_floatx(
    'float64'
)


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
        self.lr1=lr1
        self.lr2=lr2
        self.n = 0# count the number of iterations
        self.logfile=logfile
        self.logfile.write("L \t lambda \n")
        self.logfile.flush()
        self.simlogfile=simlogfile
        self.gamma = self.real_CMDP.gamma
        self.d = self.real_CMDP.d
        lbda=np.zeros(len(self.d)) + np.log(50.0)
        self.lbda = tf.Variable(lbda,dtype=np.float64)
        self.entropy_reg_constant = 5.0# by default, can override it

    def train(self):
        for t in range(self.train_iterations):
            print("start with real samples")
            time.sleep(2)
            if self.real_iterations > 0: # narrow the uncertainty set while you are going over real samples
                self.real_samples()
            print("start offline optimisation")
            time.sleep(2)
            self.offline_optimisation()

    def test(self,deterministic):
       self.real_CMDP.episode(self.pi,test=deterministic)

    def real_samples(self):
        for it in range(self.real_iterations):
            print("it ",it)
            trajectory = self.real_CMDP.episode(self.pi,test=False)
            self.uncertainty_set.add_visits(trajectory)
        self.uncertainty_set.set_params()
    def update_policy(self,V, C, d, probs, grad, eta1,eta2):
        grad_p,grad_H = grad
        if self.optimiser_lbda is not None:
            actual_lbda = tf.clip_by_value(tf.exp(self.lbda), clip_value_min=0, clip_value_max=500)
            L = -(V - K.sum(actual_lbda * C)) # dL/dtheta (min_theta L)
            update = [eta1 * ((L * g) - self.entropy_reg_constant*g_H) for (g,g_H) in zip(grad_p,grad_H)]
            self.update_theta(update)
            update_l = -eta2 * (C - d)  # dL/d\lambda  (max_lambda L)
            self.update_lbda(update_l)
        else:
            L = -V
            update = [eta1 * ((L * g) - self.entropy_reg_constant*g_H) for (g,g_H) in zip(grad_p,grad_H)]
            self.update_theta(update)
            actual_lbda=None
        return L, actual_lbda

    def update(self,trajectory,gamma,d):
        V = 0
        C = 0
        next_L = 0
        self.n += 1  # count
        eta1 = self.lr1(self.n)
        eta2 = self.lr2(self.n)
        self.uncertainty_set.count = self.n
        for i, step in enumerate(trajectory[::-1]):
            s, a, r, c, s_next, grad, probs, grad_adv, probs_adv = step
            V = r + gamma * V
            C = c + gamma * C
            L,actual_lbda = self.update_policy(V,C,d,probs,grad,eta1,eta2)
            if self.uncertainty_set.adversarial:  # min L s.t. ||P-\hat{P}|| <= alpha
                # try to make the agent fail the objective
                L_adv, lbda_adv, delta_P, alpha = self.uncertainty_set.update_adversary(eta1, eta2, s, a, next_L, grad_adv, probs_adv)
                self.logfile.write(
                        '%s \t %s \t %s \t %s \t %s \t %s \n' % (str(K.eval(next_L)), str(K.eval(actual_lbda)),
                                                                     str(L_adv), str(lbda_adv), str(delta_P), str(alpha)))
                self.logfile.flush()
                next_L = L
                #actual lbda lags one step but converges anyway
                # else: nothing to update for the last step
            elif self.uncertainty_set.critic:
                # add info to the critic
                self.uncertainty_set.update_critic(s, C)
                self.logfile.write(
                    "%.4f \t %s  \n" % (K.eval(L), str(K.eval(actual_lbda))))
                self.logfile.flush()
            else:
                self.logfile.write("%.4f \t %s \n" % (K.eval(L), str(K.eval(actual_lbda))))
                self.logfile.flush()
            # print("L",L)
            # print("lbda",actual_lbda)
        if self.uncertainty_set.critic:
            l = actual_lbda if self.uncertainty_set.using_lbda else None
            self.uncertainty_set.train_critic(l)

        if PRINTING:
            print("value ", V)
            print("cost ", C)
            print("budget ", d)

    def offline_optimisation(self):
        self.sim_CMDP = RobustCMDP.from_CMDP(self.real_CMDP,self.simlogfile,self.uncertainty_set)
        for it in range(self.sim_iterations):
            print("it ",it)
            trajectory = self.sim_CMDP.episode(self.pi,test=False)
            self.update(trajectory,self.sim_CMDP.gamma,self.sim_CMDP.d)

        print("offline optimisation ended")
        self.pi.summary()

    def update_theta(self,grad_theta):
        self.optimiser_theta.apply_gradients(zip(grad_theta, self.pi.params()))  # increase iteration by one

    def update_lbda(self,grad_lbda):
        self.optimiser_lbda.apply_gradients([(grad_lbda, self.lbda)])  # increase iteration by one

    def load(self,loadfile):
        self.pi.load(loadfile+"/pol")
        lbda=pickle.load(open(loadfile+"/objects.pkl","rb"))
        self.lbda = tf.Variable(lbda,dtype=np.float64)
        self.uncertainty_set.load(loadfile)
        print("lambda ", K.eval(self.lbda))
    def save(self,loadfile):
        self.pi.save(loadfile+"/pol")
        self.uncertainty_set.save(loadfile)
        lbda=K.eval(self.lbda)
        print("lambda ",lbda)
        pickle.dump(lbda,open(loadfile+"/objects.pkl","wb"))



