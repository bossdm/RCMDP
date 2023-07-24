import random
import sys, os
sys.path.append(os.path.abspath(".."))
from RCMDP.State import InitialDiscreteState
from RCMDP.Choose_Method import *
import os
import argparse
import csv
import numpy as np

parser = argparse.ArgumentParser(
                    prog = 'Simple maze',
                    description = 'run RL on a simple maze problem')
parser.add_argument('--m', dest='method_name',type=str,default="AdversarialRCPG_Hoeffding")
parser.add_argument('--lr',dest="learning_rate",type=float,default=0.001)
parser.add_argument('--lr2',dest="learning_rate2",type=float,default=0.0001)
parser.add_argument('--lr3',dest="learning_rate3",type=float,default=0.001)
parser.add_argument('--lr4',dest="learning_rate4",type=float,default=0.0001)
parser.add_argument('--folder',dest="folder",type=str,default="LogsAdversarialRCPG")
parser.add_argument('--run',dest="run",type=int,default=0)
parser.add_argument('--real_its',dest="real_its",type=int,default=10)
args = parser.parse_args()

if __name__ == "__main__":
    np.random.seed(args.run)
    args.folder+="/run"+str(args.run)
    #d=200   --> budget way too big so solution has C(theta) - d < 0 (inactive constraint) --> see that lbda converges to 0
    d=[4]
    T=200
    D_S=2  #(x,y) coordinates
    D_A=4
    D_C=len(d)
    #S=25 # 5 by 5 grid
    #A=4  # up,down,right, or left
    actions=[[-1,0],[1,0],[0,1],[0,-1]]
    initial_states=[[0,0]]
    p_0 = InitialDiscreteState(initial_states,probs=[1.])
    terminals=[[4,4]]
    def r_real(s_next):    # try to reach the
        x,y = s_next
        #s_next = [np.clip(x,0,4),np.clip(y,0,4)]
        return -1.0  # go to the goal location as quickly as possible (-8 is optimal)

    costly_cells = [(1,y) for y in range(4)] + [(3,2),(3,3),(3,4)]
    def c_real(s_next):  # try to reach the
        x, y = s_next
        xx = np.clip(x ,0,4)
        yy = np.clip(y,0,4)
        return np.array([1.0],dtype=float) if (xx,yy) in costly_cells else np.array([0.0],dtype=float)  # go vertical first rather than horizontal first

    def P_real(successprob,delta=None):
        def P(s,a):
            x, y = s
            r= random.random()
            if r < successprob:
                s_next = [np.clip(x + a[0], 0, 4), np.clip(y + a[1],0,4)]
            else:
                if delta is None:
                    dx=0
                    dy=0
                else:
                    dx, dy = delta[s,a]
                s_next = [np.clip(x+dx, 0, 4), np.clip(y+dy,0,4)]
            return s_next
        return P
    gamma=0.99
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
        print("created new folder ",args.folder)

    states=[[i,j] for i in range(5) for j in range(5)]
    next_states=[[0,0]] + actions    # relative state encoding
    realcmdp_logfile = open(args.folder + "/real_cmdp_log.txt", "w")
    real_cmdp = CMDP(p_0,r_real,c_real,P_real(0.80),states,actions,next_states,gamma,T,d,terminals,realcmdp_logfile)
    tests=[]
    for prob in [0.6,0.7,0.8,0.9,1.0]:
        tests.append(CMDP(p_0,r_real,c_real,P_real(prob),states,actions,next_states,gamma,T,d,terminals,realcmdp_logfile))

    pi = StochasticPol(D_S,D_A)
    sim_iterations = 5000
    real_iterations = args.real_its
    train_iterations = 1
    test_its = 50
    gamma = 0.99

    method = choose_method(args.method_name,args.learning_rate,args.learning_rate2,args.learning_rate3, args.learning_rate4,
                           args.folder,D_S,D_A,D_C,pi, real_cmdp, sim_iterations, real_iterations,
                    train_iterations,use_offset=True)
    method.train()

    # P_success test stochastic
    for t in tests:
        method.real_CMDP = t
        for i in range(test_its):
            method.test(False)
    test_values=[]
    test_constraints=[]
    lines = list(csv.reader(open(args.folder + "/real_cmdp_log.txt", 'r'), delimiter='\t'))
    for line in lines[-len(tests)*test_its:]:
        x,y=line
        test_values.append(float(x))
        test_constraints.append(float(y))
    testperformancefile = open(args.folder + "/test_performance_stochastic.txt", "w")
    testperformancefile.write("%.4f \t %.4f \t %.4f \t %.4f \n"%(np.mean(test_values),np.std(test_values),np.mean(test_constraints),np.std(test_constraints)))

    # P_success test deterministic
    for t in tests:
        method.real_CMDP = t
        for i in range(test_its):
            method.test(True)
    test_values=[]
    test_constraints=[]
    lines = list(csv.reader(open(args.folder + "/real_cmdp_log.txt", 'r'), delimiter='\t'))
    for line in lines[-len(tests)*test_its:]:
        x,y=line
        test_values.append(float(x))
        test_constraints.append(float(y))
    testperformancefile = open(args.folder + "/test_performance_deterministic.txt", "w")
    testperformancefile.write("%.4f \t %.4f \t %.4f \t %.4f \n"%(np.mean(test_values),np.std(test_values),np.mean(test_constraints),np.std(test_constraints)))


    # random perturb test stochastic
    np.random.seed(args.run)
    perturb_tests = []
    distortions = [1,2,5,10,20]
    state_actions=[(s,a) for s,state in enumerate(states) for a,action in enumerate(actions)]
    for N in distortions:
        # distort with low probability (don't want to change every state-action pair, that would be too severe)
        for it in range(test_its):
            idx=np.random.choice(range(len(state_actions)), (N,), replace=False)
            delta=np.zeros((len(states),len(actions)))
            for id in idx:
                s,a = state_actions[id]
                delta[s,a] = -1 + 2*np.random.randint(0,2) # {-1,1}
            perturb_tests.append(CMDP(p_0,r_real,c_real, P_real(0.80,delta),states,actions,next_states,gamma,T,d,terminals,realcmdp_logfile))
    for t in perturb_tests:
        method.real_CMDP = t
        method.test(False)
    test_values=[]
    test_constraints=[]
    lines = list(csv.reader(open(args.folder + "/real_cmdp_log.txt", 'r'), delimiter='\t'))
    for line in lines[-len(tests)*test_its:]:
        x,y=line
        test_values.append(float(x))
        test_constraints.append(float(y))
    testperformancefile = open(args.folder + "/perturb_test_performance_stochastic.txt", "w")
    testperformancefile.write("%.4f \t %.4f \t %.4f \t %.4f \n"%(np.mean(test_values),np.std(test_values),np.mean(test_constraints),np.std(test_constraints)))

    # test deterministic
    for t in perturb_tests:
        method.real_CMDP = t
        method.test(True)
    test_values=[]
    test_constraints=[]
    lines = list(csv.reader(open(args.folder + "/real_cmdp_log.txt", 'r'), delimiter='\t'))
    for line in lines[-len(tests)*test_its:]:
        x,y=line
        test_values.append(float(x))
        test_constraints.append(float(y))
    testperformancefile = open(args.folder + "/perturb_test_performance_deterministic.txt", "w")
    testperformancefile.write("%.4f \t %.4f \t %.4f \t %.4f \n"%(np.mean(test_values),np.std(test_values),np.mean(test_constraints),np.std(test_constraints)))
