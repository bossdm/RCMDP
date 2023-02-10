
from keras.optimizers import SGD
from RCPG import *
from LR_Schedule import *


def choose_method(method_name,alpha1,alpha2,folder,D_S,D_A,pi, real_cmdp, sim_iterations, real_iterations,
                    train_iterations):
    logfile = open(folder + "/optimisation_log.txt", "w")
    simlogfile=open(folder+"/simcmdp_log.txt","w")
    if method_name=="CPG":
        actions=[i for i in range(len(real_cmdp.actions))]
        uncertainty_set=BaseUncertaintySet(states=real_cmdp.states,actions=actions,next_states=real_cmdp.next_states)
        opt = SGD(learning_rate=alpha1)  # note: learning rate here is further multiplied by the functions above
        opt2 = SGD(learning_rate=alpha2)  # note: learning rate here is further multiplied by the functions above
        method = RCPG(pi, real_cmdp, uncertainty_set, opt, opt2, sim_iterations, real_iterations,
                    train_iterations, lr1=lr_proportional, lr2=lr_sixfifths, logfile=logfile,simlogfile=simlogfile)
    elif method_name=="RCPG_Hoeffding":
        actions = [i for i in range(len(real_cmdp.actions))]
        opt = SGD(learning_rate=alpha1)  # note: learning rate here is further multiplied by the functions above
        opt2 = SGD(learning_rate=alpha2)  # note: learning rate here is further multiplied by the functions above
        opt_adv = SGD(learning_rate=alpha1)  # note: learning rate here is further multiplied by the functions above
        opt2_adv = SGD(learning_rate=alpha2)  # note: learning rate here is further multiplied by the functions above
        uncertainty_set = HoeffdingSet(delta=0.999, states=real_cmdp.states, actions=actions, next_states=real_cmdp.next_states,
                                       D_S=D_S, D_A=D_A,
                                       optimiser_theta=opt_adv, optimiser_lbda=opt2_adv,centroids=[],
                                       writefile=folder+"/uncertaintyset")
        method = RCPG(pi, real_cmdp, uncertainty_set, opt, opt2, sim_iterations, real_iterations,
                    train_iterations, lr1=lr_proportional, lr2=lr_sixfifths,logfile=logfile,simlogfile=simlogfile)
    else:
        raise Exception("method name " +str(method_name)+ " not yet supported")

    return method