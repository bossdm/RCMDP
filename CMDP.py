"""
Class for running a CMDP
"""
class BaseCMDP(object):
    def __init__(self, states, actions, gamma,d):
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.d = d
    def episode(self,s, pi,test):
        R = 0
        C = 0
        trajectory = []
        for t in range(self.T):
            if s in self.terminals:
                break
            (s, a, r, c, s_next, grad,probs,grad_adv,probs_adv) = self.step(s,pi,test)
            trajectory.append((s, a, r, c, s_next, grad, probs,grad_adv,probs_adv))
            s = s_next
            R += r
            C += c
        self.logfile.write("%.4f \t %.4f \n"%(R,C))
        self.logfile.flush()
        return trajectory

class CMDP(BaseCMDP):
    def __init__(self,p_0,r,c,P,states,actions,next_states,gamma,T,d,terminals,logfile):
        BaseCMDP.__init__(self,states,actions,gamma,d)
        self.p_0=p_0
        self.r = r
        self.c = c
        self.P = P
        self.T = T
        self.terminals=terminals
        self.logfile=logfile
        self.logfile.write("R \t C \n")
        self.next_states=next_states # used to initalise uncertainty set's outcomes

    def episode(self,pi,test):
        s = self.p_0.generate()
        self.episode(s, pi,test)

    def step(self,s,pi,test):
        a_index, grad,probs = pi.select_action(s,deterministic=test)
        a = self.actions[a_index]
        s_next = self.P(s, a)
        c = self.c(s_next)
        r = self.r(s_next)
        # print("s_next ",s_next)
        return (s, a, r, c, s_next,grad,probs,None,None)

class RobustCMDP(CMDP):
    def __init__(self,p_0,r,c,P,states,actions,next_states,gamma,T,d,terminals,logfile,uncertainty_set):
        CMDP.__init__(self,p_0,r,c,P,states,actions,next_states,gamma,T,d,terminals,logfile)
        self.uncertainty_set = uncertainty_set
    @classmethod
    def from_CMDP(cls,cmdp,logfile,uncertainty_set):
        rcmdp = RobustCMDP(cmdp.p_0,cmdp.r,cmdp.c,cmdp.P,cmdp.states,cmdp.actions,cmdp.next_states,cmdp.gamma,cmdp.T,cmdp.d,cmdp.terminals,logfile,uncertainty_set)
        return rcmdp
    def step(self,s,pi,test):
        s_index = self.uncertainty_set.states.index(s)
        a_index, grad, probs = pi.select_action(s,deterministic=test)
        s_next, grad_adv, probs_adv = self.uncertainty_set.random_state(s_index, a_index)
        c = self.c(s_next)
        r = self.r(s_next)
        return (s, a_index, r, c, s_next, grad, probs, grad_adv, probs_adv)