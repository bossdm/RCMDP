

from RCMDP.State import InitialDiscreteState
from RCMDP.Choose_Method import *
import os
import argparse
import csv

parser = argparse.ArgumentParser(
                    prog = 'Simple maze',
                    description = 'run RL on a simple maze problem')
parser.add_argument('--m', dest='method_name',type=str)
parser.add_argument('--lr',dest="learning_rate",type=float)
parser.add_argument('--lr2',dest="learning_rate2",type=float)
parser.add_argument('--folder',dest="folder",type=str)
args = parser.parse_args()

if __name__ == "__main__":
    args.method_name="AdversarialRCPG_Hoeffding"
    args.learning_rate=0.10
    args.learning_rate2=0.01
    args.folder="/home/david/PycharmProjects/RCMDP/Logs/"
    #d=200   --> budget way too big so solution has C(theta) - d < 0 (inactive constraint) --> see that lbda converges to 0
    d=[1]
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

    def P_real(s, a):  # try to reach the
        x, y = s
        s_next = [np.clip(x + a[0], 0, 4), np.clip(y + a[1],0,4)]
        return s_next   # go vertical first rather than horizontal first
    gamma=0.99
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
        print("created new folder ",args.folder)

    states=[[i,j] for i in range(5) for j in range(5)]
    next_states=states
    realcmdp_logfile = open(args.folder + "/real_cmdp_log.txt", "w")
    real_cmdp = CMDP(p_0,r_real,c_real,P_real,states,actions,next_states,gamma,T,d,terminals,realcmdp_logfile)

    pi = StochasticPol(D_S,D_A)
    sim_iterations = 1000
    real_iterations = 100
    train_iterations = 1
    test_its = 100
    gamma = 0.99

    method = choose_method(args.method_name,args.learning_rate,args.learning_rate2,args.folder,D_S,D_A,D_C,pi, real_cmdp, sim_iterations, real_iterations,
                    train_iterations)
    method.train()
    for i in range(test_its):
        method.test()
    test_values=[]
    test_constraints=[]
    lines = list(csv.reader(open(args.folder + "/real_cmdp_log.txt", 'r'), delimiter='\t'))
    for line in lines[-test_its:]:
        x,y=line
        test_values.append(float(x))
        test_constraints.append(float(y))
    testperformancefile = open(args.folder + "/test_performance.txt", "w")
    testperformancefile.write("%.4f \t %.4f \t %.4f \t %.4f \n"%(np.mean(test_values),np.std(test_values),np.mean(test_constraints),np.std(test_constraints)))