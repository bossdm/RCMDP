
import numpy as np
from RCMDP.Policy import StochasticPol
from RCMDP.Critic import Critic
import tensorflow as tf
import keras.backend as K
import pickle
from RCMDP.Utils import check_folder
from collections import OrderedDict
import time
PRINTING=True
N_NORM_SAMPLES=100

class BaseUncertaintySet(object):
    adversarial = False
    critic = False
    def __init__(self,states,actions,next_states,centroids=None,use_offset=False,writefile=None):
        self.S=len(states)
        self.A=len(actions)
        self.NS=len(next_states)
        self.data = np.zeros((self.S,self.A,self.NS)) + 1  # SxAxS frequency table  (1 pseudo count to make all states being visited at least ones and have non-zero prob)
        self.states=states
        self.actions=actions
        self.next_states = next_states
        self.nominal=np.zeros((self.S,self.A,self.NS)) + 1./self.NS # uniform at the start
        self.centroids=centroids
        self.use_offset=use_offset
        state_matrix = np.array(states)
        self.s_min = np.min(state_matrix,axis=0)
        self.s_max = np.max(state_matrix,axis=0)
        self.writefile=writefile
        self.U_updates=0


    def get_closest(self,state):

        minDist=float("inf")
        minIndex=None
        for i,s in enumerate(self.states):
            s = np.array(s)
            d = np.sum(np.square(state - s))
            if d < minDist:
                minDist =d
                minIndex = i
        return minIndex
    def offset_state(self,s_next,s):
        return [s_next[i] - s[i] for i in range(len(s))]
    def add_visits(self,trajectory):
        for s, a_index, _r, _c, s_next, _grad,_probs,_grad_adv,_probs in trajectory:
            if self.centroids:
                s_index = self.get_closest(s)
            else:
                s_index = self.states.index(s)
            if self.use_offset:
                ss = self.offset_state(s_next,s)
                s_next_index = self.next_states.index(ss)
            else:
                s_next_index = self.next_states.index(s_next)
            self.data[s_index, a_index, s_next_index] += 1  # add trajectory to the data counts

    def set_params(self):

        if self.writefile is not None:
            f = open(self.writefile+str(self.U_updates)+".txt","w")
            self.U_updates += 1
        self.visits = np.sum(self.data,
                             axis=2)  # sum over third axis (don't care about next state, only the sa-visitations
        for s_index in range(self.S):
            for a_index in range(self.A):
                if self.visits[s_index, a_index] > 0: #otherwise keep at uniform random
                    self.nominal[s_index, a_index] = self.data[s_index, a_index] / self.visits[s_index, a_index]
                f.write("%d \t %d \t %d " % (
                s_index, a_index, self.visits[s_index, a_index]))
                for s_next_index in range(self.NS):
                    f.write("\t %.4f" % (self.nominal[s_index, a_index, s_next_index]))
                f.write("\n")
        f.close()
    def random_state(self,s,a):
        try:
            s_next_index = np.random.choice(self.NS,p=self.nominal[s,a])
        except Exception as e:
            print(self.nominal[s,a])
            print(e)
        s_next=self.next_states[s_next_index]
        return s_next, None, None

    def save(self,loadfile):
        folder=loadfile+"/uncertaintyset/"
        check_folder(folder)
        objects=(self.S,self.A,self.NS,self.data,self.states,self.actions,self.next_states,self.nominal,self.centroids)
        pickle.dump(objects,open(loadfile+"/uncertaintyset.pkl","wb"))
    def load(self,loadfile):
        folder=loadfile+"/uncertaintyset/"
        check_folder(folder)
        self.S,self.A,self.NS,self.data,self.states,self.actions,\
            self.next_states,self.nominal,self.centroids=pickle.load(open(loadfile+"/uncertaintyset.pkl","rb"))

class HoeffdingSet(BaseUncertaintySet):

    def __init__(self,critic_type,CI, states,actions, next_states,D_S, D_A,D_C,centroids=None,use_offset=False,writefile=None):
        BaseUncertaintySet.__init__(self,states,actions,next_states,centroids,use_offset,writefile)
        self.critic = True
        self.adversarial = False
        self.delta = 1 - CI # set delta according to desired confidence level
        self.alpha=np.zeros((self.S,self.A))
        self.D_S = D_S
        self.D_A = D_A
        self.D_C = D_C
        self.batch = []
        self.critic_type = critic_type
        if self.critic_type == "V":
            self.V = Critic(self.D_S,1)
        elif self.critic_type == "C":
            self.C = Critic(self.D_S,self.D_C)
        elif self.critic_type == "L":
            self.V = Critic(self.D_S, 1)
            self.C = Critic(self.D_S, self.D_C)
        else:
            raise Exception("critic type "+self.critic_type+" not found. Use either 'C', 'V', or 'L'")
        self.set_cumulativecost(lbda=None) # initialisation never uses lambda

    def set_cumulativecost(self,lbda): # keep dict of cost of states in exactly the order as the states occur and are indexed
        self.cumulativecost_dict = OrderedDict([(i,self.cumulativecost(state,lbda)) for i,state in enumerate(self.states)])
    def cumulativecost(self,state,lbda):
        """
        predict cumulative cost(s) from the given state.
        if a single cost, then direct prediction from the critic network
        if multiple costs, these are reduced to a single number by; a) a weighted sum of the Lagrangian multiplers is taken;
        or b) a simple sum
        """

        if self.critic_type == "V":
            Costs = - self.V.predict([state])[0]
        elif self.critic_type == "C":
            Costs = self.C.predict([state])[0]
            if lbda is None:
                lbda = np.ones(len(Costs))
        elif self.critic_type == "L":
            V = - self.V.predict([state])[0]
            Costs = self.C.predict([state])[0]
            Costs = np.vstack((V,Costs))
            if lbda is None:
                lbda = np.ones(len(Costs))
            else:
                lbda = np.vstack((np.array([1.0]),lbda))
        else:
            raise Exception("critic type "+self.critic_type+" not found. Use either 'C', 'V', or 'L'")

        if len(Costs) == 1:
            s = Costs[0]
        elif len(Costs) > 1:
            s = np.sum(lbda*Costs)
        else:
            raise Exception("no cost vector should have zero length")
        if PRINTING:
            print("lbda ",lbda)
            print("Critic costs ", Costs)
            print("summed costs ", s)
        return s
    def update_cost_critic(self,x,y):
        # if PRINTING:
        #     print("x", x)
        #     print("y", y)
        self.C.add_to_batch(x,y)
    def update_value_critic(self,x,y):
        # if PRINTING:
        #     print("x", x)
        #     print("y", y)
        self.V.add_to_batch(x,y)
    def train_critic(self,lbda):
        if self.critic_type == "V":
            self.V.train()
        elif self.critic_type == "C":
            self.C.train()
        else:
            self.V.train()
            self.C.train()

        self.set_cumulativecost(lbda)
    def add_visits(self,trajectory):
        for s,a_index,_r,_c,s_next,_grad,_probs,_grad_adv,_probs in trajectory:
            if self.centroids:
                s_index = self.get_closest(s)
            else:
                s_index = self.states.index(s)      # note: this uncertainty set always assumes the same state space for s and s_next
            if self.use_offset:
                ss = self.offset_state(s_next,s)
                s_next_index = self.next_states.index(ss)
            else:
                s_next_index = self.next_states.index(s_next)
            self.data[s_index,a_index,s_next_index] += 1   # add trajectory to the data counts
    def set_params(self):
        """

        :param
        :return:
        """
        if self.writefile is not None:
            f = open(self.writefile+str(self.U_updates)+".txt","w")
            self.U_updates+=1
        self.visits = np.sum(self.data,axis=2) # sum over third axis (don't care about next state, only the sa-visitations
        for s_index in range(self.S):
            for a_index in range(self.A):
                if self.visits[s_index, a_index] > 0: #otherwise keep at uniform random
                    self.nominal[s_index, a_index] = self.data[s_index, a_index] / self.visits[s_index, a_index]
                    self.alpha[s_index,a_index] = self.compute_alpha(s_index,a_index)
                if self.writefile is not None:
                        f.write("%d \t %d \t %d \t %.4f"%(s_index,a_index,self.visits[s_index,a_index],self.alpha[s_index,a_index]))
                        for s_next_index in range(self.NS):
                            f.write("\t %.4f"%(self.nominal[s_index, a_index,s_next_index]))
                        f.write("\n")
        f.close()
    def get_best_index(self,sorted_idxs,s):
        state = self.states[s]
        for i in sorted_idxs:
            next_state = self.states[i]
            next_state = self.offset_state(next_state, state)
            if next_state in self.next_states:
                return self.next_states.index(next_state)
        raise Exception("index not found!")
    def get_worst_index(self,start,sorted_idxs,s):
        state = self.states[s]
        for i in range(start,-1,-1):
            k = sorted_idxs[i]
            next_state = self.states[k]
            next_state = self.offset_state(next_state, state)
            if next_state in self.next_states:
                return self.next_states.index(next_state)
        raise Exception("index not found!")
    def solve_innerproblem(self,s,a):  #solve inner problem following
        sorted_idxs = sorted(self.cumulativecost_dict, key=self.cumulativecost_dict.get,reverse=True)

        if self.use_offset:
            k = self.get_best_index(sorted_idxs, s)
        else:
            k = sorted_idxs[0]

        P_hat_k = self.nominal[s,a,k]
        eps = min(self.alpha[s,a] / 2, 1 - self.nominal[s,a,k])
        P = [p for p in self.nominal[s,a]]
        P[k] = P_hat_k + eps  # add to probability mass to state with the highest cost; then compensate by removing elsewhere
        i = len(sorted_idxs) - 1 # start subtracting from lowest cost indexes
        while eps > 0: # while added mass remains to be subtracted
            if self.use_offset:
                k = self.get_worst_index(i,sorted_idxs,s)
            else:
                k = sorted_idxs[i]
            sub = min(eps, P[k])
            P[k] -= sub
            eps -= sub
            i -= 1
        # if PRINTING:
        #     print("costs ", self.cumulativecost_dict)
        #     print("nominal ", self.nominal[s,a])
        #     print("P ", P)
        return P
    def random_state(self,s,a):
        probs=self.solve_innerproblem(s,a)
        s_next_index = np.random.choice(self.NS,p=probs)
        s_next = self.next_states[s_next_index]
        return s_next,None,probs

    def compute_alpha(self,s,a):
        """
        :return:
        """
        return np.sqrt(2/self.visits[s,a]*np.log(self.S*self.A*2**self.NS/self.delta))


    def save(self,loadfile):
        folder=loadfile+"/uncertaintyset/"
        check_folder(folder)
        objects=(self.S,self.A,self.NS,self.data,self.states,self.actions,self.next_states,self.nominal,self.centroids,
                 self.delta,self.alpha,self.D_S,self.D_A,self.U_updates)
        #print("objects:")
        #print(objects)
        pickle.dump(objects,open(folder+"objects.pkl","wb"))
        if self.critic_type == "V":
            self.V.save(folder+"critic_V")
        elif self.critic_type == "C":
            self.C.save(folder+"critic_C")
        else:
            self.C.save(folder+"critic_C")
            self.V.save(folder+"critic_V")

    def load(self,loadfile):
        folder=loadfile+"/uncertaintyset/"
        check_folder(folder)
        objects=pickle.load(open(folder+"objects.pkl","rb"))
        self.S, self.A, self.NS, self.data, self.states, self.actions, self.next_states, self.nominal, self.centroids, \
            self.delta, self.alpha, self.D_S, self.D_A, self.U_updates = objects
       
        if self.critic_type == "V":
            self.V.load(folder+"critic_V")
        elif self.critic_type == "C":
            self.C.load(folder+"critic_C")
        else:
            self.C.load(folder+"critic_C")
            self.V.load(folder+"critic_V")
        #print("objects:")
        #print(objects)



class AdversarialHoeffdingSet(BaseUncertaintySet):
    """
    uncertainty set based on Hoeffding; adds adversarial agent to solve the inner problem approximately
    """

    def __init__(self,CI, states,actions, next_states,D_S, D_A, optimiser_theta,optimiser_lbda,centroids=None,
                 use_offset=False, writefile=None,min_count=50):
        BaseUncertaintySet.__init__(self,states,actions,next_states,centroids,use_offset)
        self.delta = 1 - CI # set delta according to desired confidence level
        self.alpha=np.zeros((self.S,self.A))
        self.D_S = D_S
        self.D_A = D_A
        self.pi = StochasticPol(self.D_S+1,self.NS) # +1 is for actions; S is output as we want output probs for each state
        self.optimiser_theta = optimiser_theta
        self.optimiser_lbda  = optimiser_lbda
        self.lbda = tf.Variable(np.log(1.0),dtype=np.float64)  # norm constraint is always active
        self.writefile=writefile
        self.U_updates=0
        self.adversarial = True
        self.critic = False
        self.count = 0
        self.min_count = min_count
        self.approximating_done = False

    def add_visits(self,trajectory):
        for s,a_index,_r,_c,s_next,_grad,_probs,grad_adv,_probs in trajectory:
            if self.centroids:
                s_index = self.get_closest(s)
            else:
                s_index = self.states.index(s)
            if self.use_offset:
                ss = self.offset_state(s_next,s)
                s_next_index = self.next_states.index(ss)
            else:
                s_next_index = self.next_states.index(s_next)
            self.data[s_index,a_index,s_next_index] += 1   # add trajectory to the data counts
    def set_params(self):
        """

        :param
        :return:
        """
        if self.writefile is not None:
            f = open(self.writefile+str(self.U_updates)+".txt","w")
            self.U_updates+=1
        self.visits = np.sum(self.data,axis=2) # sum over third axis (don't care about next state, only the sa-visitations
        for s_index in range(self.S):
            for a_index in range(self.A):
                if self.visits[s_index, a_index] > 0: #otherwise keep at uniform random
                    self.nominal[s_index, a_index] = self.data[s_index, a_index] / self.visits[s_index, a_index]
                    self.alpha[s_index,a_index] = self.compute_alpha(s_index,a_index)
                if self.writefile is not None:
                        f.write("%d \t %d \t %d \t %.4f"%(s_index,a_index,self.visits[s_index,a_index],self.alpha[s_index,a_index]))
                        for s_next_index in range(self.NS):
                            f.write("\t %.4f"%(self.nominal[s_index, a_index,s_next_index]))
                        f.write("\n")
        f.close()

    def get_adversary_state(self,s,a):
        l = np.array([s/self.S for s in self.states[s]] + [self.actions[a]/self.A])
        return l
    def random_probs(self,s,a):
        """
        :param s:
        :param a:
        :return:
        """
        adversary_state = self.get_adversary_state(s,a)
        if self.count < self.min_count - 1:
            probs = np.random.choice(self.NS,p=self.nominal[s,a])
            return probs, (None,None,None,None), None
        s_next_index, grad, probs = self.pi.select_action(adversary_state,deterministic=False,nominal=self.nominal[s,a])
        return s_next_index,grad,probs

    def random_state(self,s,a):
        s_next_index,grad,probs=self.random_probs(s,a)
        s_next = self.next_states[s_next_index]
        return s_next,grad,probs

    def compute_alpha(self,s,a):
        """
        :return:
        """
        return np.sqrt(2/self.visits[s,a]*np.log(self.S*self.A*2**self.NS/self.delta))
    def nominal_approximate(self):
        print("start training nominal approximator")
        lossfile = open("lossfile.txt", "w")
        for epoch in range(10000):
            data_x = []
            data_y = []
            for s in range(self.S):
                for a in range(self.A):
                    data_x.append(self.get_adversary_state(s, a))
                    data_y.append(self.nominal[s, a])
            data_x = np.array(data_x)
            data_y = np.array(data_y)
            with tf.GradientTape() as tape:
                y = self.pi.predict(data_x)
                loss = tf.keras.losses.MeanAbsoluteError()(data_y, y)
                grads = tape.gradient(loss, self.pi.params())
            self.optimiser_theta.apply_gradients(zip(grads, self.pi.params()))
            lossfile.write(str(K.eval(loss)) + "\n")
            self.approximating_done = True

    def compute_nominal_deviation_grad(self):
        data_x = []
        alpha = []
        nominal = []
        states = np.random.random_integers(0,self.S-1,N_NORM_SAMPLES)
        actions = np.random.random_integers(0, self.A - 1, N_NORM_SAMPLES)
        for s,a in zip(states,actions):
                data_x.append(self.get_adversary_state(s, a))
                alpha.append(self.alpha[s,a])
                nominal.append(self.nominal[s, a])
        data_x = np.array(data_x)
        nominal = np.array(nominal)
        alpha = np.array(alpha)
        with tf.GradientTape() as tape:
            y = self.pi.predict(data_x)
            deviation = tf.norm(y - nominal , ord=1) - alpha
            norms = tf.math.maximum(deviation,0)   # penalise any occurence of norm > alpha
            loss = tf.reduce_mean(norms)
            grads = tape.gradient(loss, self.pi.params())
        return grads

    def update_adversary(self,eta1,eta2,s,a,L,grad_adv,probs):
        """
        adversary to make the optimisation of the lagrangian as difficult as possible;
        min L s.t. \Delta P \leq alpha
        :param eta1: learning rate
        :param L: lagrangian of the actual CMDP
        :param grad_adv: gradient of the adversarial policy
        :param delta_P: ||P - P_nominal||
        :return:
        """
        if self.count ==1 and not self.approximating_done:
            loss = self.nominal_approximate()
            return 0,0,loss,0
        elif self.count < self.min_count:
            return 0,0,0,0
        if self.centroids:
            s_index = self.get_closest(s)
        else:
            s_index = self.states.index(s)

        grad_theta, grad_H,delta_P = grad_adv # grad_P,
        grad_P = self.compute_nominal_deviation_grad()
        actual_lbda = tf.clip_by_value(tf.exp(self.lbda), clip_value_min=0, clip_value_max=500)
        update = [eta1*(L*grad + actual_lbda *grad_P[g]) for g,grad in enumerate(grad_theta)]  #dL/dtheta_adv : min_thetaadv L_adv
        self.optimiser_theta.apply_gradients(zip(update, self.pi.params()))  # increase iteration by one
        update_l = -(eta2 * (delta_P - self.alpha[s_index,a]))  # dL/d\lambda  (max_lambda  L_adv)
        self.optimiser_lbda.apply_gradients([(update_l, self.lbda)])  # increase iteration by one
        L_adv = L + actual_lbda * (delta_P - self.alpha[s_index,a]) # keep track of L_adv
        return K.eval(L_adv), K.eval(actual_lbda), K.eval(delta_P), self.alpha[s_index,a]
        #print("L_adv ", L_adv)
        #print("lbda_adv", self.lbda)

    def save(self,loadfile):
        folder=loadfile+"/uncertaintyset/"
        check_folder(folder)
        objects=(self.S,self.A,self.NS,self.data,self.states,self.actions,self.next_states,self.nominal,self.centroids,
                 self.delta,self.alpha,self.D_S,self.D_A, K.eval(self.lbda),self.U_updates)
        #print("objects:")
        #print(objects)
        pickle.dump(objects,open(folder+"objects.pkl","wb"))
        self.pi.save(folder+"adversary")
    def load(self,loadfile):
        folder=loadfile+"/uncertaintyset/"
        check_folder(folder)
        objects=pickle.load(open(folder+"objects.pkl","rb"))
        self.S, self.A, self.NS, self.data, self.states, self.actions, self.next_states, self.nominal, self.centroids, \
            self.delta, self.alpha, self.D_S, self.D_A, lbda, self.U_updates = objects
        #print("objects:")
        #print(objects)
        self.lbda.assign(lbda)
        self.pi.load(folder+"adversary")

#
# class BayesOptSet(object):
#     """
#     uncertainty set for even larger/continuous state spaces; adds adversarial agent to solve the inner problem approximately
#     """
#     adversarial = True
#     def __init__(self,delta, states,actions, D_S, D_A, optimiser_theta,optimiser_lbda,GP_params):
#         self.delta = delta # desired confidence level
#         self.S=len(states)
#         self.A=len(actions)
#         self.data = np.zeros((self.S,self.A,self.S))  # SxAxS frequency table
#         self.states=states
#         self.actions=actions
#         self.nominal=np.zeros((self.S,self.A,self.S)) + 1/self.S # uniform at the start
#         self.alpha=np.zeros((self.S,self.A))
#         self.D_S = D_S
#         self.D_A = D_A
#         self.pi = StochasticPol(self.D_S+1,self.S) # +1 is for actions; S is output as we want output probs for each state
#         self.optimiser_theta = optimiser_theta
#         self.optimiser_lbda  = optimiser_lbda
#         self.lbda = tf.Variable(0.1)
#         self.GP = GP(GP_params)
#
#     def add_visits(self,trajectory):
#         for s,a,_r,_c,s_next,_grad,_grad_adv,_probs in trajectory:
#             self.GP.add([s,a],s_next)
#     def set_params(self):
#         """
#
#         :param
#         :return:
#         """
#         pass
#     def random_probs(self,s,a):
#         """
#         :param s:
#         :param a:
#         :return:
#         """
#         adversary_state=self.states[s]+(self.actions[a],)
#         a, grad, probs = self.pi.select_action(adversary_state,deterministic=False)
#         return probs,grad
#     def compute_alpha(self,sd):
#         """
#         :return:
#         """
#         return
#     def random_state(self,s,a):
#         probs,grad=self.random_probs(s,a)
#         s_next_index = np.random.choice(self.S,p=probs)
#         s_next=self.states[s_next_index]
#         return s_next,grad,probs
#
#     def update_adversary(self,eta1,eta2,s,a,L,grad_adv,probs):
#         """
#         adversary to make the optimisation of the lagrangian as difficult as possible;
#         min L s.t. \Delta P \leq alpha
#         :param eta1: learning rate
#         :param L: lagrangian of the actual CMDP
#         :param grad_adv: gradient of the adversarial policy
#         :param delta_P: ||P - P_nominal||
#         :return:
#         """
#         M,S = self.GP.predict([s,a])
#         nominal = M
#         alpha = S #TODO is this the best?
#         delta_P = np.sum(np.abs(probs - nominal))
#         L_adv = L - self.lbda*delta_P   # minimise the original lagrangian s.t. constraints on the norm
#         update = [eta1*L_adv*g for g in grad_adv]
#         self.optimiser_theta.apply_gradients(zip(update, self.pi.params()))  # increase iteration by one
#         update_l = -eta2 * (delta_P - alpha)  # dL/d\lambda
#         self.optimiser_lbda.apply_gradients([(update_l, self.lbda)])  # increase iteration by one
#         return K.eval(L_adv), K.eval(self.lbda)
#         #print("L_adv ", L_adv)
#         #print("lbda_adv", self.lbda)


#
#
# class BCIUncertaintySet(object):   # see algorithm 2 in appendix of BCI paper
#     """
#     given data set D, BCI solves the optimisation problem for tight ambiguity sets:
#
#     min {phi: P(|| \bar{P}_s,a - P^*_s,a || > phi | D ) < delta / SA
#     """
#     def __init__(self,posterior,delta,n):
#         self.posterior=posterior
#         self.delta=delta
#         self.n = n
#         self.nominal = np.zeros((self.S,self.A,self.S)) + 1/self.S # uniform at the start
#         self.alpha = np.zeros((self.S, self.A))
#
#     def add_visits(self, trajectory):
#         self.posterior.update(trajectory)
#     def set_params(self):
#         for s_index in range(self.S):
#             for a_index in range(self.S):
#                 probslist = []
#                 for i in range(self.n):
#                     probs = self.posterior.generate(s_index,a_index)
#                     probslist.append(probs)
#                 self.nominal[s_index,a_index] = np.mean(probslist)
#                 dists=np.sum(np.abs(np.array(probslist) - self.nominal[s_index,a_index]),axis=1)
#                 sorted_dists = np.sort(dists)[::-1] # sorted in increasing order
#                 idx = self.delta*self.n
#                 self.alpha[s_index,a_index] = sorted_dists[idx]
#
# class AdversarialSet(object):
#     """
#     works against the agent, i.e. minimises the lagrangian by selecting a next state for a given state-action pair
#     """
#     adversarial=True
#     def __init__(self, states, D_S):
#         self.states=states
#         self.D_S = D_S
#         self.pi = StochasticPol(self.D_S+1, self.D_S)  # + 1 for the actions
#
#     def add_visits(self, trajectory):
#         for s_index, a_index, _r, _c, s_next, _grad in trajectory:
#             s_next_index = self.states.index(s_next)
#             self.data[s_index, a_index, s_next_index] += 1  # add trajectory to the data counts
#
#     def set_params(self):
#         self.visits = np.sum(self.data,
#                              axis=2)  # sum over third axis (don't care about next state, only the sa-visitations
#         for s_index in range(self.S):
#             for a_index in range(self.A):
#                 if self.visits[s_index, a_index] > 0:  # otherwise keep at uniform random
#                     self.nominal[s_index, a_index] = self.data[s_index, a_index] / self.visits[s_index, a_index]
#
#     def random_state(self, s, a):
#         try:
#             s_next_index = np.random.choice(self.S, p=self.nominal[s, a])
#         except Exception as e:
#             print(self.nominal[s, a])
#             print(e)
#         s_next = self.states[s_next_index]
#         return s_next

if __name__ == "__main__":
    PRINTING=True
    U = HoeffdingSet(0.90,states=[1,2,3,4],actions=[1,2],next_states=[1,2,3,4],D_S=1,D_A=1)
    U.cumulativecost_dict=OrderedDict([(i,i*0.1) for i,state in enumerate(U.states)])
    for s,state in enumerate(U.states):
        for a, action in enumerate(U.actions):
            U.alpha[s,a] = 0.20
            U.nominal[s,a] = [np.random.random() for s in U.next_states]
            U.nominal[s,a] /= sum(U.nominal[s,a])
            U.solve_innerproblem(s,a)
