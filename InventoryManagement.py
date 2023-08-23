"""
the classic inventory management problem (Zipkin, 2000; Puterman, 2005 p. 37--40).
state s_t is the inventory/stock
action a_t is the number of units ordered.
demand D_t has time-homogeneous probability distribution.

transition dynamics: s_{t+1} = max( s_t + a - D, 0)
reward: r(s_t,a_t,s_{t+1}) = - O(a_t) - h(s_t + a_t) + f(s_t + a_t - s_{t+1})


where O(x) = K + c(x) represents the purchase costs
      h(x) = x  represents the holding costs
      f(x) =          represents the revenue


for convenience compute r(s_t,a_t)  =  sum_{s_{t+1}} p(s_{t+1}|s_t,a_t) r(s_t,a_t,s_{t+1})

--> r(s,a) = F(s+a) - O(a) - h(s+a)  # revenue - ordering costs - holding costs

the number of states, S, is the maximum of the inventory.
this also means  actions M-s+1,..., M are not available (or give zero probability to them)


parameters are set as implemented in

Behzadian, B., Russel, R., & Petrik, M. (2019). High-Confidence Policy Optimization: Reshaping Ambiguity Sets in Robust MDPs. 1–17.

"The inventory level is discrete and limited by the number of states S.
The purchase cost K, sale price, and holding cost are 2.49, 3.99, and 0.03 respectively.
The demand is sampled from a normal distribution with a mean S/4 and a standard deviation of S/6.
The initial state is 0 (empty stock)."
"""
import argparse
import itertools

import numpy as np
from scipy.stats import norm

# import sys,os
# RCMDP_path=os.environ["BENCHMARK_DIR"]+"/RCMDP"
# sys.path.extend([RCMDP_path])
from RCMDP.CMDP import BaseCMDP
from RCMDP.Utils import *
from RCMDP.agent.agent_env_loop import agent_env_loop
from RCMDP.agent.set_agent import *

parser = argparse.ArgumentParser(
                    prog = 'Inventory Management',
                    description = 'run inventory management benchmark')
parser.add_argument('--m', dest='method_name',type=str, default="PG")
parser.add_argument('--lr',dest="learning_rate",type=float,default=0.001)  # remember annealing reduces this
parser.add_argument('--lr2',dest="learning_rate2",type=float,default=0.0001)
parser.add_argument('--lr3',dest="learning_rate3",type=float,default=0.001)  # remember annealing reduces this
parser.add_argument('--lr4',dest="learning_rate4",type=float,default=0.0001)
parser.add_argument('--r',dest="run",type=int,default=0)
parser.add_argument('--folder',dest="folder",type=str,default=os.environ["RESULTS_DIR"])
parser.add_argument('--saving',dest="saving_freq",type=int,default=1000)
parser.add_argument('--real_its',dest="real_its",type=int,default=10)
parser.add_argument('--run',dest="run",type=int,default=0)
parser.add_argument('--stage',dest="stage",type=str,default="data")

args = parser.parse_args()

PRINTING=False
if PRINTING:
    printfile=open("IM_log.txt","w")
    printfile.write("s \t a \t r \t c  \t inventory \t demand \t s_next \n")
class InventoryManagement(BaseCMDP):
    def __init__(self,gamma,d,S,using_nextstate):
        states=[[s] for s in range(S)]
        actions=[[a] for a in range(S)]
        BaseCMDP.__init__(self,None,states,actions,gamma,None,d,None,None)  #p_0,states, actions, gamma,T,d,terminals,logfile
        self.next_states=states
        self.S = S
        self.D_S = 1
        self.D_A = self.S
        self.purchase_cost = 2.49
        self.sale_price = 3.99
        self.holding_cost = 0.03
        self.mu_dem = self.S / 4
        self.sigma_dem = self.S / 6.
        self.purchase_limit = self.mu_dem + self.sigma_dem  # soft purchase limit
        self.stepsPerEpisode=100
        self.using_nextstate = using_nextstate
    def perturb(self,mu_factor,sigma_factor):
        self.mu_dem = mu_factor*self.S
        self.sigma_dem = sigma_factor*self.S
    def demand(self):
        self.D = int(round(max(0,np.random.normal(self.mu_dem,self.sigma_dem))))

    def transition(self,s,a):
        self.demand()
        return max(s + a -self.D, 0)

    def get_demand_probabilities(self,inventory):  # inventory = s + a
        demand_probabilities = []
        cum=0
        for demand in range(inventory): # practical maximum for the demand given our parameters
            new_cum = norm.cdf(demand,loc=self.mu_dem, scale=self.sigma_dem)  # e.g. if demand is inventory - 1, then s_next = 1 (inserted at the first index)
            demand_probabilities.insert(0,new_cum - cum)
            cum = new_cum   # note that the last loop (j=inventory represents the
        # s_next = 0 at the zeroth index, is represented by 1 - cum (the remaining probability; all cases with demand greater or equal to the inventory)
        demand_probabilities.insert(0,1 - cum)
        return demand_probabilities

    def expected_revenue(self,inventory):   # if you know the demand probabilities
        ps = self.get_demand_probabilities(inventory)
        r = 0
        for demand, p in enumerate(ps[:-1]):
            r+=p*self.f(demand)
        r+=ps[-1]*self.f(inventory) # can only sell as much as you have in stock
        return r
    def f(self,sold):
        return self.sale_price*sold
    def revenue(self,s,a,s_next): # if you do not know demand probabilities (robust)
        return self.f(s+a-s_next)

    def ordering_costs(self,order):
        return self.purchase_cost*order

    def holding_costs(self,inventory):
        return inventory*self.holding_cost

    def r(self,s,a):
        return self.expected_revenue(s+a) - self.ordering_costs(a) - self.holding_costs(s+a)

    def c(self,s,a): # supplier limits
        if s <= 2:
            L = self.purchase_limit
        else:
            L = self.mu_dem 
        return np.array([max(0,float(a) - L )])

    def actual_action(self,s,a):
        return min(a, self.S - s - 1) # order can be at most the remaining room in inventory
    def step(self,message,repeat_steps=0):
        s = self.s_index
        a=message[0]
        a = self.actual_action(s,a)
        if self.stage == "train":   # use the nextstate to set the demand (either the nominal or a worst-case model but not the real one)
            self.D = self.mu_dem
            s_next = self.next_state
        else:
            self.demand()
            s_next = [self.transition(s, a)]

        #else:
            # next state is determined only by uncertainty set


        #delta, grad_adv, probs_adv = self.uncertainty_set.random_state(self.s_index, a)
        c = self.c(s,a)
        r = self.r(s,a)
        if PRINTING:
            printfile.write("%d \t %d \t %.4f \t %d \t %d \t %s \t %d \n"%(s,a,r,c[0],s+a,str(self.D),s_next[0]))
        self.s_index = s_next[0]
        return (s_next, r, c, False, [])

    def reset(self):
        self.s_index = 0
        self.episodeScore = 0
        self.episodeConstraint = np.zeros(len(self.d))
        return [self.s_index]  # starting state is zero: empty inventory
    def solved(self):
        return False
    def get_true_d0(self):
        d0 = np.zeros(self.S)
        d0[0] = 1 # S=0 is always the starting state
        return d0
    def get_true_P(self):
        P = np.zeros((len(self.states),len(self.actions),len(self.states)+1))
        for s in range(len(self.states)):  # go over all non-terminal states
            for a in range(len(self.actions)):
                actual_action = self.actual_action(s,a)
                probs=self.get_demand_probabilities(s +actual_action)
                for s_next in range(len(probs)):
                    P[s,a,s_next] = probs[s_next]   # note: s_next is never greater than s+a (so expect zeros at the end)
        return P
    def get_true_R(self):
        R = np.zeros((len(self.states),len(self.actions),len(self.states)+1)) - 1 # default reward is - 1
        # using (s,a) formalism so reward is the same regardless of next state
        for s in range(len(self.states)):  # go over all non-terminal states
            for a in range(len(self.actions)):
                actual_action = self.actual_action(s,a)
                R[s,a,:] = np.zeros(len(self.states)+1) + self.r(s,actual_action)  # note: there is no terminal state so last state has probability zero
        return R
    def candidate_statesets(self):
        nonterminal_states=range(len(self.states))
        return list(itertools.combinations(nonterminal_states, 0)) + list(itertools.combinations(nonterminal_states, 1)) + \
                  list(itertools.combinations(nonterminal_states, 2))

if __name__ == "__main__":
    args.folder+="/run"+str(args.run)
    check_folder(args.folder+ "/models_and_plots/")
    gamma =0.99
    S = 10
    d = np.array([6.0])   # sum(0.99**i for i in range(100)) * 0.1  (supplier limits can only be exceeded a fraction of the time or for small  quantity)
    # corresponds to episodeConstraint ~= 10.0 (not discounted)
    end_real_experiences = args.real_its
    end_simulated_experiences = end_real_experiences + 5000
    test_its = 50
    # for tests manipulate μ ∈ {S/6, S/4, S/3} and σ ∈ {S/8, S/6, S/4}
    mu_factors=[]
    sigma_factors=[]
    for mu_factor in [1./6.,0.25,1./3.]:
        for sigma_factor in [0.125,1./6.,0.25]:
            mu_factors.append(mu_factor)
            sigma_factors.append(sigma_factor)
    N_tests = len(mu_factors)

    using_nextstate=args.stage=="train"

    np.random.seed(args.run)
    

    # first real samples
    env = InventoryManagement(gamma, d, S, using_nextstate)
    if args.stage=="data":
        env.stage="data"
        print("start real samples to get nominal dynamics")
        agent = set_agent(args, env)
        agent_env_loop(env, agent, args, episodeCount=0, episodeLimit=end_real_experiences, using_nextstate=using_nextstate)
        print("nominal dynamics:", agent.uncertainty_set.nominal)
        # #print("uncertainty budget:", agent.uncertainty_set.alpha)
    elif args.stage=="train":  # do train and test in 1 run
        # # now simulate using the estimated dynamics
        env.stage="train"
        print("start training")
        agent = set_agent(args, env)
        resume(args.folder+ "/models_and_plots/",agent)
        agent_env_loop(env, agent, args, episodeCount=end_real_experiences, episodeLimit=end_simulated_experiences,
                       using_nextstate=using_nextstate)
        print("agent trained")
    elif args.stage=="test_stoch" or args.stage=="test_determ":
        # now test using the real samples again
        #trajectories=pickle.load(open("trajectories_"+str(args.run)+".pkl","rb"))
        end_last = end_simulated_experiences
        env.stage = "test"
        determ=args.stage=="test_determ"
        agent = set_agent(args, env)
        resume(args.folder + "/models_and_plots/", agent)
        for t in range(N_tests):
            end_test = end_simulated_experiences + test_its * (t+1)
            env.perturb(mu_factors[t],sigma_factors[t])
            print("testing the agent on real samples")
            agent_env_loop(env, agent, args, episodeCount=end_last, episodeLimit=end_test,
                           using_nextstate=using_nextstate,deterministic=determ)
            print("#tests ", end_test - end_simulated_experiences)
            #pickle.dump(agent.trajectories,open("trajectories_"+str(args.run)+".pkl","wb"))
            end_last = end_test
    else:
        raise Exception("stage "+args.stage  + " not recognised")
