
from keras.optimizers import Adam
from RCMDP.RCPG import *
from RCMDP.LR_Schedule import *


def choose_method(method_name,alpha1,alpha2,alpha3,alpha4,folder,D_S,D_A,D_C,pi, real_cmdp, sim_iterations, real_iterations,
                    train_iterations,use_offset):
    logfile = open(folder + "/optimisation_log.txt", "w")
    simlogfile=open(folder+"/simcmdp_log.txt","w")
    if method_name=="RCPG_Hoeffding_C":
        actions = [i for i in range(len(real_cmdp.actions))]
        opt = Adam(learning_rate=alpha1)  # note: learning rate here is further multiplied by the functions above
        opt2 = Adam(learning_rate=alpha2)  # note: learning rate here is further multiplied by the functions above
        uncertainty_set = HoeffdingSet(critic_type="C",delta=0.9, states=real_cmdp.states, actions=actions, next_states=real_cmdp.next_states,
                                       D_S=D_S, D_A=D_A,D_C=D_C,centroids=None,use_offset=use_offset,
                                       writefile=folder+"/uncertaintyset")
        method = RCPG(pi, real_cmdp, uncertainty_set, opt, opt2, sim_iterations, real_iterations,
                    train_iterations, lr1=lr_proportional, lr2=lr_proportional,logfile=logfile,simlogfile=simlogfile)
    elif method_name == "RCPG_Hoeffding_V":
            actions = [i for i in range(len(real_cmdp.actions))]
            opt = Adam(learning_rate=alpha1)  # note: learning rate here is further multiplied by the functions above
            opt2 = Adam(learning_rate=alpha2)  # note: learning rate here is further multiplied by the functions above
            uncertainty_set = HoeffdingSet(critic_type="V", delta=0.9, states=real_cmdp.states, actions=actions,
                                           next_states=real_cmdp.next_states,
                                           D_S=D_S, D_A=D_A,D_C=D_C, centroids=None,use_offset=use_offset,
                                           writefile=folder + "/uncertaintyset")
            method = RCPG(pi, real_cmdp, uncertainty_set, opt, opt2, sim_iterations, real_iterations,
                          train_iterations, lr1=lr_proportional, lr2=lr_proportional, logfile=logfile,
                          simlogfile=simlogfile)
    elif method_name == "RCPG_Hoeffding_L":
            actions = [i for i in range(len(real_cmdp.actions))]
            opt = Adam(learning_rate=alpha1)  # note: learning rate here is further multiplied by the functions above
            opt2 = Adam(learning_rate=alpha2)  # note: learning rate here is further multiplied by the functions above
            uncertainty_set = HoeffdingSet(critic_type="L",delta=0.9, states=real_cmdp.states, actions=actions,
                                           next_states=real_cmdp.next_states,
                                           D_S=D_S, D_A=D_A,D_C=D_C, centroids=[],use_offset=use_offset,
                                           writefile=folder + "/uncertaintyset")
            method = RCPG(pi, real_cmdp, uncertainty_set, opt, opt2, sim_iterations, real_iterations,
                          train_iterations, lr1=lr_proportional, lr2=lr_proportional, logfile=logfile,
                          simlogfile=simlogfile)
    elif method_name=="CPG":
        actions=[i for i in range(len(real_cmdp.actions))]
        uncertainty_set=BaseUncertaintySet(states=real_cmdp.states,actions=actions,next_states=real_cmdp.next_states,use_offset=use_offset)
        opt = Adam(learning_rate=alpha1)  # note: learning rate here is further multiplied by the functions above
        opt2 = Adam(learning_rate=alpha2)  # note: learning rate here is further multiplied by the functions above
        method = RCPG(pi, real_cmdp, uncertainty_set, opt, opt2, sim_iterations, real_iterations,
                    train_iterations, lr1=lr_proportional, lr2=lr_proportional, logfile=logfile,simlogfile=simlogfile)
    elif method_name=="PG":
        actions=[i for i in range(len(real_cmdp.actions))]
        uncertainty_set=BaseUncertaintySet(states=real_cmdp.states,actions=actions,next_states=real_cmdp.next_states,use_offset=use_offset)
        opt = Adam(learning_rate=alpha1)  # note: learning rate here is further multiplied by the functions above
        opt2 = None  # no constraints so no second optimiser
        method = RCPG(pi, real_cmdp, uncertainty_set, opt, opt2, sim_iterations, real_iterations,
                    train_iterations, lr1=lr_proportional, lr2=lr_proportional, logfile=logfile,simlogfile=simlogfile)
    elif method_name=="AdversarialRCPG_Hoeffding":
        actions = [i for i in range(len(real_cmdp.actions))]
        opt = Adam(learning_rate=alpha1)  # note: learning rate here is further multiplied by the functions above
        opt2 = Adam(learning_rate=alpha2)  # note: learning rate here is further multiplied by the functions above
        opt_adv = Adam(learning_rate=alpha3)  # note: learning rate here is further multiplied by the functions above
        opt2_adv = Adam(learning_rate=alpha4)  # note: learning rate here is further multiplied by the functions above
        uncertainty_set = AdversarialHoeffdingSet(delta=0.9, states=real_cmdp.states, actions=actions, next_states=real_cmdp.next_states,
                                       D_S=D_S, D_A=D_A,
                                       optimiser_theta=opt_adv, optimiser_lbda=opt2_adv,centroids=None,use_offset=use_offset,
                                       writefile=folder+"/uncertaintyset")
        method = RCPG(pi, real_cmdp, uncertainty_set, opt, opt2, sim_iterations, real_iterations,
                    train_iterations, lr1=lr_proportional, lr2=lr_proportional,logfile=logfile,simlogfile=simlogfile)
    elif method_name=="random":
        actions=[i for i in range(len(real_cmdp.actions))]
        method=BaseUncertaintySet(states=real_cmdp.states,actions=actions,next_states=real_cmdp.next_states)
    elif method_name=="random_hoeffding":
        actions = [i for i in range(len(real_cmdp.actions))]
        method = HoeffdingSet(critic_type="V",delta=0.9, states=real_cmdp.states, actions=actions, next_states=real_cmdp.next_states,
                                       D_S=D_S, D_A=D_A,D_C=D_C,centroids=None,
                                       writefile=folder+"/uncertaintyset")

    else:
        raise Exception("method name " +str(method_name)+ " not yet supported")

    return method
