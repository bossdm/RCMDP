
import numpy as np
from RCMDP.Utils import check_folder
def after_episode(env,agent,step,actionProbsList,saving_frequency,episodeCount,folder):
    if env.stage == "data":
        # update the agent uncertainty set
        agent.uncertainty_set.add_visits(agent.buffer)
        agent.uncertainty_set.set_params()
        env.uncertainty_set = agent.uncertainty_set
    elif env.stage == "train":
        maxActionProb = np.max(actionProbsList)
        # averageEpisodeActionProbs.append(avgActionProb)
        print("Max action prob:", maxActionProb)
        agent.trainStep(batchSize=step + 1)  # train the agent
        if (episodeCount % saving_frequency) == 0:
            print("saving at episode count " + str(episodeCount))
            agent.save(folder+"/models_and_plots/", episodeCount)
        solved = env.solved()  # Check whether the task is solved
    elif env.stage == "test":  # "test"
        # nothing to update
        print("continue testing ")
    else:
        raise Exception("environment stage should be either data, train, or test. Got " + env.stage + "instead")
    env.episodeScoreList.append(env.episodeScore)
    env.episodeConstraintList.append(env.episodeConstraint)
def after_loop(env,agent,folder,episodeCount,solved):
    if env.stage == "training": # this was a training session
        agent.save(folder+"/models_and_plots/",episodeCount)
        if not solved:
            print("Reached episode limit and task was not solved, deploying agent for testing...")
        else:
            print("Task is solved, deploying agent for testing...")
    check_folder(folder+"/performance/")
    writefile = open(folder+"/performance/"+env.stage+".txt","w")
    for i in range(len(env.episodeScoreList)):
        writefile.write("%.4f "%(env.episodeScoreList[i],))
        for j in range(len(env.d)):
            writefile.write("\t %.4f" % (env.episodeConstraintList[i][j],))
        writefile.write("\n")
    print("avg score ", np.mean(env.episodeScoreList))
    print("avg constraint ", np.mean(env.episodeConstraintList))

def agent_env_loop(env,agent,args,episodeCount,episodeLimit,using_nextstate=False):

    env.uncertainty_set = agent.uncertainty_set
    #env.uncertainty_set.centroids = env.states
    saving_frequency=args.saving_freq
    solved = False  # Whether the solved requirement is met
    env.episodeScoreList = []
    env.episodeConstraintList = []
    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount <= episodeLimit:
        s = env.reset()  # Reset robot and get starting observation
        print("start episode at state ",s)
        actionProbsList = []  # This list holds the probability of each chosen action
        # Inner loop is the episode loop
        for step in range(env.stepsPerEpisode):
            # # In training mode the agent samples from the probability distribution, naturally implementing exploration
            a, grad, actionProbs = agent.work(s, test=False)
            env.next_state, env.grad_adv, env.probs_adv = agent.random_state(env.s_index, a)
            s_next, r, c, done, info = env.step([a],repeat_steps=1)
            if using_nextstate:
                trans = (s, a, r, c, env.next_state, grad, actionProbs, env.grad_adv, env.probs_adv) # use the offset
            else:
                trans = (s, a, r, c, s_next, grad, actionProbs,env.grad_adv, env.probs_adv)
            # note that RCPG does not technically use s_next in update of networks but it does in add_visits to update the uncertainty set;
            # but in general useful to store for other learners (e.g. experience replay)
            agent.storeTransition(trans)
            actionProbsList.append(actionProbs)
            env.episodeScore += r  # Accumulate episode reward
            env.episodeConstraint += c  # Accumulate episode constraint
            if done:
                break

            s = s_next  # state for next step is current step's newState

        print("Episode #", episodeCount, "score:", env.episodeScore)
        print("Episode #", episodeCount, "constraint:", env.episodeConstraint)
        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.

        after_episode(env, agent, step, actionProbsList, saving_frequency, episodeCount, args.folder)
        episodeCount += 1  # Increment episode counter

    after_loop(env,agent,args.folder,episodeCount,solved)
