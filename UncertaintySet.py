
import numpy as np
from Policy import StochasticPol
import tensorflow as tf
import keras.backend as K
import pickle
from Utils import check_folder

class BaseUncertaintySet(object):
    adversarial = False
    def __init__(self,states,actions,next_states,centroids=[]):
        self.S=len(states)
        self.A=len(actions)
        self.NS=len(next_states)
        self.data = np.zeros((self.S,self.A,self.NS))  # SxAxS frequency table
        self.states=states
        self.actions=actions
        self.next_states = next_states
        self.nominal=np.zeros((self.S,self.A,self.NS)) + 1/self.NS # uniform at the start
        self.centroids=centroids

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

    def add_visits(self,trajectory):
        for s, a_index, _r, _c, s_next_index, _grad,_probs,_grad_adv,_probs in trajectory:
            if self.centroids:
                s_index = self.get_closest(s)
            else:
                s_index = self.states.index(s)
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
        self.S,self.A,self.NS,self.data,self.states,self.actions,self.next_states,self.nominal,self.centroids=pickle.load(open(loadfile+"/uncertaintyset.pkl","wb"))


class HoeffdingSet(BaseUncertaintySet):
    """
    uncertainty set based on Hoeffding; adds adversarial agent to solve the inner problem approximately
    """
    adversarial = True
    def __init__(self,delta, states,actions, next_states,D_S, D_A, optimiser_theta,optimiser_lbda,centroids=[],writefile=None):
        BaseUncertaintySet.__init__(self,states,actions,next_states,centroids)
        self.delta = delta # desired confidence level
        self.alpha=np.zeros((self.S,self.A))
        self.D_S = D_S
        self.D_A = D_A
        self.pi = StochasticPol(self.D_S+1,self.NS) # +1 is for actions; S is output as we want output probs for each state
        self.optimiser_theta = optimiser_theta
        self.optimiser_lbda  = optimiser_lbda
        self.lbda = tf.Variable(0.1)
        self.writefile=writefile
        self.U_updates=0

    def add_visits(self,trajectory):
        for s,a_index,_r,_c,s_next,_grad,_probs,_grad_adv,_probs in trajectory:
            if self.centroids:
                s_index = self.get_closest(s)
            else:
                s_index = self.states.index(s)
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

    def random_probs(self,s,a):
        """
        :param s:
        :param a:
        :return:
        """
        adversary_state = list(self.states[s]) + [self.actions[a]]
        a, grad, probs = self.pi.select_action(adversary_state,deterministic=False)
        return probs,grad

    def random_state(self,s,a):
        probs,grad=self.random_probs(s,a)
        s_next_index = np.random.choice(self.NS,p=probs)
        s_next = self.next_states[s_next_index]
        return s_next,grad,probs

    def compute_alpha(self,s,a):
        """
        :return:
        """
        return np.sqrt(2/self.visits[s,a]*np.log(self.S*self.A*2**self.NS/self.delta))

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
        if self.centroids:
            s_index = self.get_closest(s)
        else:
            s_index = self.states.index(s)
        delta_P = np.sum(np.abs(probs - self.nominal[s_index, a]))
        L_adv = L - self.lbda*delta_P   # minimise the original lagrangian s.t. constraints on the norm
        update = [eta1*L_adv*g for g in grad_adv]
        self.optimiser_theta.apply_gradients(zip(update, self.pi.params()))  # increase iteration by one
        update_l = -eta2 * (delta_P - self.alpha[s_index,a])  # dL/d\lambda
        self.optimiser_lbda.apply_gradients([(update_l, self.lbda)])  # increase iteration by one
        return K.eval(L_adv), K.eval(self.lbda)
        #print("L_adv ", L_adv)
        #print("lbda_adv", self.lbda)

    def save(self,loadfile):
        folder=loadfile+"/uncertaintyset/"
        check_folder(folder)
        objects=(self.S,self.A,self.NS,self.data,self.states,self.actions,self.next_states,self.nominal,self.centroids,
                 self.delta,self.alpha,self.D_S,self.D_A, K.eval(self.lbda),self.U_updates)
        print("objects:")
        print(objects)
        pickle.dump(objects,open(folder+"objects.pkl","wb"))
        self.pi.save(folder+"adversary")
    def load(self,loadfile):
        folder=loadfile+"/uncertaintyset/"
        check_folder(folder)
        objects=pickle.load(open(folder+"objects.pkl","rb"))
        self.S, self.A, self.NS, self.data, self.states, self.actions, self.next_states, self.nominal, self.centroids, \
            self.delta, self.alpha, self.D_S, self.D_A, lbda, self.U_updates = objects
        print("objects:")
        print(objects)
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
