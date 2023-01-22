"""
Class for running a CMDP
"""

class CMDP(object):
    def __init__(self,p_0,r,c,P,actions,T,terminals):
        self.p_0=p_0
        self.r = r
        self.c = c
        self.P = P
        self.actions=actions
        self.T=T
        self.terminals=terminals

    def step(self,s,pi):
        a_index, grad = pi.select_action(s)
        a = self.actions[a_index]
        s_next = self.P(s, a)
        c = self.c(s_next)
        r = self.r(s_next)
        # print("s_next ",s_next)
        return (s, a, r, c, s_next,grad)

    def episode(self,pi):
        R = 0
        C = 0
        s = self.p_0.generate()
        trajectory = []
        for t in range(self.T):
            if s in self.terminals or t > 200:
                break
            (s, a, r, c, s_next, grad) = self.step(s,pi)
            trajectory.append((s, a, r, c, s_next, grad))
            s = s_next
            R += r
            C += c
        print("R", R)
        print("C", C)
        return trajectory,t

class RobustCMDP(CMDP):
    def __init__(self,p_0,r,c,P,actions,T,terminals,uncertainty_set):
        CMDP.__init__(self,p_0,r,c,P,actions,T,terminals)
        self.uncertainty_set = uncertainty_set
    @classmethod
    def from_CMDP(cls,cmdp,uncertainty_set):
        rcmdp = RobustCMDP(cmdp.p_0,cmdp.r,cmdp.c,cmdp.P,cmdp.actions,cmdp.T,cmdp.terminals,uncertainty_set)
        return rcmdp
    def step(self,s,pi):
        s_index = self.uncertainty_set.states.index(s)
        a_index, grad = pi.select_action(s)
        s_next = self.uncertainty_set.random_state(s_index, a_index)
        c = self.c(s_next)
        r = self.r(s_next)
        return (s_index, a_index, r, c, s_next, grad)