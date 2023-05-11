import random
import sys, os
sys.path.append(os.path.abspath(".."))
from RCMDP.State import InitialDiscreteState
from RCMDP.Choose_Method import *
import os
import argparse
import csv

parser = argparse.ArgumentParser(
                    prog = 'Simple maze',
                    description = 'run RL on a simple maze problem')
parser.add_argument('--m', dest='method_name',type=str,default="AdversarialRCPG_Hoeffding")
parser.add_argument('--lr',dest="learning_rate",type=float,default=0.001)
parser.add_argument('--lr2',dest="learning_rate2",type=float,default=0.0001)
parser.add_argument('--lr3',dest="learning_rate3",type=float,default=0.001)
parser.add_argument('--lr4',dest="learning_rate4",type=float,default=0.0001)
parser.add_argument('--folder',dest="folder",type=str,default="LogsAdversarialRCPG")
args = parser.parse_args()

if __name__ == "__main__":
    #d=200   --> budget way too big so solution has C(theta) - d < 0 (inactive constraint) --> see that lbda converges to 0
    d=[4]
    T=200
    D_S=2  #(x,y) coordinates
    D_A=4
    D_C=len(d)
    #S=25 # 5 by 5 grid
    #A=4  # up,down,right, or left
    actions=[[-1,0],[1,0],[0,1],[0,-1]]
    # triangle_states=[(x,y) for y in range(4) for x in range(4) if x <= y]
    # l=len(triangle_states)
    initial_states=[[0,0]]
    p_0 = InitialDiscreteState(initial_states,probs=[1.])
    terminals=[[4,4]]
    def r_real(s_next):    # try to reach the
        x,y = s_next
        s_next = [np.clip(x,0,4),np.clip(y,0,4)]
        return 1.0 if s_next == [4,4] else -1.0  # go to the goal location

    def c_real(s_next):  # try to reach the
        x, y = s_next
        xx = np.clip(x ,0,4)
        yy = np.clip(y,0,4)
        return np.array([1.0],dtype=float) if xx - yy > 1 else np.array([0.0],dtype=float)  # go vertical first rather than horizontal first

    def P_real(successprob):
        def P(s,a):
            x, y = s
            r= random.random()
            if r < successprob:
                s_next = [np.clip(x + a[0], 0, 4), np.clip(y + a[1],0,4)]
            else:
                s_next = [np.clip(x, 0, 4), np.clip(y,0,4)]
            return s_next
        return P
    gamma=0.99
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
        print("created new folder ",args.folder)

    states=[[i,j] for i in range(5) for j in range(5)]
    next_states=[[0,0]] + actions    # relative state encoding
    realcmdp_logfile = open(args.folder + "/real_cmdp_log.txt", "w")
    real_cmdp = CMDP(p_0,r_real,c_real,P_real(0.90),states,actions,next_states,gamma,T,d,terminals,realcmdp_logfile)
    tests=[]
    for prob in [0.5,0.6,0.7,0.8,1.0]:
        tests.append(CMDP(p_0,r_real,c_real,P_real(prob),states,actions,next_states,gamma,T,d,terminals,realcmdp_logfile))

    pi = StochasticPol(D_S,D_A)
    sim_iterations = 500
    real_iterations = 10
    train_iterations = 1
    test_its = 10
    gamma = 0.99

    method = choose_method(args.method_name,args.learning_rate,args.learning_rate2,args.learning_rate3, args.learning_rate4,
                           args.folder,D_S,D_A,D_C,pi, real_cmdp, sim_iterations, real_iterations,
                    train_iterations,use_offset=True)
    method.train()
    for t in tests:
        method.real_CMDP = t
        for i in range(test_its):
            method.test()
    test_values=[]
    test_constraints=[]
    lines = list(csv.reader(open(args.folder + "/real_cmdp_log.txt", 'r'), delimiter='\t'))
    for line in lines[-len(tests)*test_its:]:
        x,y=line
        test_values.append(float(x))
        test_constraints.append(float(y))
    testperformancefile = open(args.folder + "/test_performance.txt", "w")
    testperformancefile.write("%.4f \t %.4f \t %.4f \t %.4f \n"%(np.mean(test_values),np.std(test_values),np.mean(test_constraints),np.std(test_constraints)))