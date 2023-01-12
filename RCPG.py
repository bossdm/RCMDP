""" implements a scalable version of Robust Constrained Policy Gradient (Russel et al.)
    where scalability is due to avoiding the inner problem in favour of a perturbation scheme

 """


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from State import InitialDiscreteState
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
from random import choice

class RCPG(object):
    def __init__(self,pi, actions, gamma, d, p_0,T,terminals,c_hat,r_hat,P_hat, perturbations,optimiser_theta,optimiser_lbda, iterations):
        """
        :param pi:  policy, a keras model
        :param actions: the action set
        :param gamma: discount factor
        :param d: constraint-cost budget
        :param p_0: initial state distribution
        :param T: limited step trajectory
        :param terminals: set of states that are terminal
        :param c_hat: estimated constraint-cost function
        :param r_hat: estimated reward function
        :param P_hat: estimated transition dynamics
        :param perturbation: the chosen perturbation scheme to the nominal dynamics
        :param optimiser_theta: the chosen optimiser for the policy
        :param optimiser_lbda: the chosen optimiser for the lagrangian multiplier
        :param iterations
        """
        self.pi = pi
        self.actions = actions
        self.gamma = gamma
        self.p_0 = p_0
        self.T = T
        self.terminals = terminals
        self.c_hat = c_hat
        self.r_hat = r_hat
        self.P_hat = P_hat
        self.perturbations = perturbations
        self.optimiser_theta = optimiser_theta
        self.optimiser_lbda = optimiser_lbda
        self.d = d
        self.iterations = iterations
        self.lbda = tf.Variable(0.1)

    def simulation_step(self,s):
        with tf.GradientTape() as tape:
            a_probs = self.pi(np.array([s]))[0]
            grad = tape.gradient(a_probs, self.pi.trainable_weights)
        a = np.random.choice(len(a_probs),p=K.eval(a_probs))
        a = self.actions[a]
        p = np.random.choice(len(self.perturbations))
        perturb = self.perturbations[p]
        c = self.c_hat(s, a, perturb)
        r = self.r_hat(s, a, perturb)
        s_next = self.P_hat(s, a, perturb)
        #print("s_next ",s_next)
        return (s,a,r,c,s_next,grad)
    def test(self):
        R=0
        s = p_0.generate()
        t=0
        while True:
            if s in self.terminals or t >200:
                break
            (s, a, r, c, s_next, grad) = self.simulation_step(s)
            s=s_next
            R+=r
            t+=1
        print("R",R)
        return R
    def offline_optimisation(self):

        trajectories = []
        for it in range(self.iterations):
            R = 0
            print("iteration ", it, "/",self.iterations)
            s = p_0.generate()
            for t in range(self.T):
                if s in self.terminals:
                    break
                (s,a,r,c,s_next,grad) = self.simulation_step(s)
                trajectories.append((s,a,r,c,s_next,grad))
                s=s_next
                R += r
            print("R", R)
            T_stop = t - 1
            V = 0
            C = 0
            for t in range(T_stop,-1,-1):
                s,a,r,c,s_next,grad = trajectories[t]
                V = r + self.gamma * V
                C = c + self.gamma * C
                actual_lbda = tf.clip_by_value(tf.exp(self.lbda), clip_value_min=0, clip_value_max=1000)
                L = -(V-actual_lbda*C)
                update = [L * g for g in grad]   # dL/d\pi * d\pi/d\theta
                self.update_theta(update)
                update_l = -(C - self.d)           # dL/d\lambda
                self.update_lbda(update_l)
                #print("L",L)
                #print("lbda",actual_lbda)
            print("L",L)
            print("lbda",actual_lbda)
        print("offline optimisation ended")
        self.pi.summary()

    def update_theta(self,grad_theta):
        """ TODO: make sure learning rate schedule implemented and connection with keras"""
        self.optimiser_theta.apply_gradients(zip(grad_theta, self.pi.trainable_weights))  # increase iteration by one


    def update_lbda(self,grad_lbda):
        """ TODO: make sure learning rate schedule implemented and connection with keras"""
        self.optimiser_lbda.apply_gradients([(grad_lbda, self.lbda)])  # increase iteration by one




if __name__ == "__main__":
    #d=200   --> budget way too big so solution has C(theta) - d < 0 (inactive constraint) --> see that lbda converges to 0
    d=1
    T=200
    D_S=2  #(x,y) coordinates
    D_A=4
    actions=[(-1,0),(1,0),(0,1),(0,-1)]
    # triangle_states=[(x,y) for y in range(4) for x in range(4) if x <= y]
    # l=len(triangle_states)
    initial_states=[(0,0)]
    p_0 = InitialDiscreteState(initial_states,probs=[1.])
    terminals=[(4,4)]
    def r_hat(s,a,perturb):    # try to reach the
        x,y = s
        s_next = (np.clip(x+a[0]+perturb[0],0,4),np.clip(y+a[1]+perturb[1],0,4))
        return 1.0 if s_next == (4,4) else -1.0  # go to the goal location

    def c_hat(s, a, perturb):  # try to reach the
        x, y = s
        xx = np.clip((x + a[0] + perturb[0]),0,4)
        yy = np.clip((y + a[1] + perturb[1]),0,4)
        return 1.0 if xx - yy > 1 else 0.0  # go vertical first rather than horizontal first

    def P_hat(s, a, perturb):  # try to reach the
        x, y = s
        s_next = (np.clip(x + a[0] +perturb[0], 0, 4), np.clip(y + a[1] + perturb[1],0,4))
        return s_next   # go vertical first rather than horizontal first
    pi = Sequential()
    pi.add(Dense(20, name="hidden1",activation="relu",kernel_initializer='uniform', input_shape=(D_S,)))
    pi.add(Dense(D_A, name="output",activation="softmax"))
    X, Y = np.mgrid[-1:2:1, -1:2:1]
    perturbations = [(0,0)]#list(np.vstack((X.flatten(), Y.flatten())).T)

    opt = Adam(learning_rate=0.10)
    opt2 = Adam(learning_rate=0.10)

    iterations = 1000
    gamma = 0.99
    rcpg = RCPG(pi, actions, gamma , d, p_0, T, terminals, c_hat, r_hat, P_hat, perturbations, opt, opt2,iterations)
    rcpg.offline_optimisation()
    rcpg.test()


