
from RCMDP.Choose_Method import choose_method
from RCMDP.agent.RCPG_Agent import RCPG_Agent
from RCMDP.agent.random_agent import RandomAgent
from RCMDP.UncertaintySet import *

def set_agent(args,env,use_offset=False):
    D_S = env.D_S
    D_A = env.D_A
    D_C = len(env.d)
    if args.method_name.startswith("AdversarialRCPG") or args.method_name.startswith("RCPG"):
        pi = StochasticPol(D_S=D_S, D_A=D_A)
        method = choose_method(args.method_name, args.learning_rate, args.learning_rate2,
                               args.learning_rate3, args.learning_rate4,args.folder,
                               D_S=D_S, D_A=D_A,D_C=D_C,
                               pi=pi, real_cmdp=env,
                               sim_iterations=None, real_iterations=None,
                               train_iterations=None,use_offset=use_offset)
        return RCPG_Agent.from_RCPG(method,no_noise=False)
    elif args.method_name.startswith("CPG"):
        pi = StochasticPol(D_S=D_S, D_A=D_A)
        method = choose_method("CPG", args.learning_rate, args.learning_rate2, args.folder, D_S=D_S, D_A=D_A,D_C=D_C,
                               pi=pi, real_cmdp=env,
                               sim_iterations=None, real_iterations=None,
                               train_iterations=None,use_offset=use_offset)
        if args.method_name=="CPG_nonoise":
            return RCPG_Agent.from_RCPG(method,no_noise=True)
        else:
            return RCPG_Agent.from_RCPG(method, no_noise=False)
    elif args.method_name.startswith("random"):
        uncertainty_set = choose_method(args.method_name, args.learning_rate, args.learning_rate2, args.folder, D_S=D_S, D_A=D_A,D_C=D_C,

                               pi=None, real_cmdp=env,
                               sim_iterations=None, real_iterations=None,
                               train_iterations=None,use_offset=use_offset)
        return RandomAgent(D_A,uncertainty_set)
    elif args.method_name=="PG":
        pi = StochasticPol(D_S=D_S, D_A=D_A)
        method = choose_method("PG", args.learning_rate, args.learning_rate2, args.folder, D_S=D_S, D_A=D_A,D_C=D_C,
                               pi=pi, real_cmdp=env,
                               sim_iterations=None, real_iterations=None,
                               train_iterations=None,use_offset=use_offset)
        return RCPG_Agent.from_RCPG(method, no_noise=False)
    else:
        raise Exception("method ", args.method_name, " not yet supported")
