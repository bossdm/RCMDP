""" implements a scalable version of Robust Constrained Policy Gradient (Russel et al.)
    where scalability is due to avoiding the inner problem in favour of a perturbation scheme

 """


import tensorflow as tf
from State import InitialDiscreteState
from keras.optimizers import SGD
import numpy as np
from Policy import StochasticPol
from UncertaintySet import *
from CMDP import *
from keras.callbacks import LearningRateScheduler

class RCPG(object):
    def __init__(self,pi, actions, gamma, d,  real_CMDP, uncertainty_set,optimiser_theta,optimiser_lbda,
                 sim_iterations,real_iterations,train_iterations,lr1,lr2):
        """
        :param pi:  policy, a keras model
        :param actions: the action set
        :param gamma: discount factor
        :param d: constraint-cost budget
        :param real_cmdp: the real CMDP
        :param uncertainty_set: the chosen uncertainty_set
        :param optimiser_theta: the chosen optimiser for the policy
        :param optimiser_lbda: the chosen optimiser for the lagrangian multiplier
        :param sim_iterations: the number of iterations for offline optimisation during the training
        :param real_iterations: the number of iterations of real trajectories during the training
        :param train_iterations: the number of training iterations
        :param callbacks_list1
        :param callbacks_list
        """
        self.pi = pi
        self.actions = actions
        self.gamma = gamma
        self.real_CMDP = real_CMDP
        self.uncertainty_set = uncertainty_set
        self.optimiser_theta = optimiser_theta
        self.optimiser_lbda = optimiser_lbda
        self.d = d
        self.sim_iterations = sim_iterations
        self.real_iterations = real_iterations
        self.train_iterations = train_iterations
        self.lbda = tf.Variable(0.1)
        self.lr1=lr1
        self.lr2=lr2
        self.n = 0# count the number of iterations
    # def real_step(self,s):
    #     a_index, grad = self.pi.select_action(s)
    #     a = self.actions[a_index]
    #     s_next = self.P(s,a)
    #     c = self.c(s_next)
    #     r = self.r(s_next)
    #     # print("s_next ",s_next)
    #     return (s, a, r, c, s_next)
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
        self.sim_CMDP = RobustCMDP.from_CMDP(self.real_CMDP,self.uncertainty_set)
        for it in range(self.sim_iterations):
            trajectory,t = self.sim_CMDP.episode(self.pi)
            T_stop = t - 1
            V = 0
            C = 0
            self.n += 1 # count
            for t in range(T_stop,-1,-1):
                s,a,r,c,s_next,grad = trajectory[t]
                V = r + self.gamma * V
                C = c + self.gamma * C
                actual_lbda = tf.clip_by_value(tf.exp(self.lbda), clip_value_min=0, clip_value_max=10000)
                eta1=self.lr1(self.n)
                eta2=self.lr2(self.n)
                L = -(V-actual_lbda*C)
                update = [eta1 * L * g for g in grad]   # dL/d\pi * d\pi/d\theta
                self.update_theta(update)
                update_l = -eta2*(C - self.d)           # dL/d\lambda
                self.update_lbda(update_l)
                # print("L",L)
                # print("lbda",actual_lbda)
            print("L",L)
            print("lbda",actual_lbda)

        print("offline optimisation ended")
        self.pi.summary()

    def update_theta(self,grad_theta):
        self.optimiser_theta.apply_gradients(zip(grad_theta, self.pi.params()))  # increase iteration by one


    def update_lbda(self,grad_lbda):
        self.optimiser_lbda.apply_gradients([(grad_lbda, self.lbda)])  # increase iteration by one




if __name__ == "__main__":
    #d=200   --> budget way too big so solution has C(theta) - d < 0 (inactive constraint) --> see that lbda converges to 0
    d=1
    T=200
    D_S=2  #(x,y) coordinates
    D_A=4
    #S=25 # 5 by 5 grid
    #A=4  # up,down,right, or left
    actions=[(-1,0),(1,0),(0,1),(0,-1)]
    # triangle_states=[(x,y) for y in range(4) for x in range(4) if x <= y]
    # l=len(triangle_states)
    initial_states=[(0,0)]
    p_0 = InitialDiscreteState(initial_states,probs=[1.])
    terminals=[(4,4)]
    def r_real(s_next):    # try to reach the
        x,y = s_next
        s_next = (np.clip(x,0,4),np.clip(y,0,4))
        return 1.0 if s_next == (4,4) else -1.0  # go to the goal location

    def c_real(s_next):  # try to reach the
        x, y = s_next
        xx = np.clip(x ,0,4)
        yy = np.clip(y,0,4)
        return 1.0 if xx - yy > 1 else 0.0  # go vertical first rather than horizontal first

    def P_real(s, a):  # try to reach the
        x, y = s
        s_next = (np.clip(x + a[0], 0, 4), np.clip(y + a[1],0,4))
        return s_next   # go vertical first rather than horizontal first
    real_cmdp = CMDP(p_0,r_real,c_real,P_real,actions,T,terminals)
    states=[(i,j) for i in range(5) for j in range(5)]
    actions=[i for i in range(4)]
    pi = StochasticPol(D_S,D_A)
    def lr1(n):
        return 1./(n)    # stochastic approximation at two time scales (see Borkar, 2009)
    def lr2(n):
        return 1/(n**1.2)
    opt = SGD(learning_rate=0.1)  # note: learning rate here is further multiplied by the functions above
    # it is set just before applying gradients based on the schedule functions above
    opt2 = SGD(learning_rate=0.01)
    sim_iterations = 1000
    real_iterations = 100
    train_iterations = 1
    gamma = 0.99
    uncertainty_set=DummyUncertaintySet(states=states,actions=actions)
    rcpg = RCPG(pi, actions, gamma , d, real_cmdp, uncertainty_set, opt, opt2, sim_iterations,real_iterations,train_iterations,lr1,lr2)
    rcpg.train()
    for i in range(100):
        rcpg.test()


