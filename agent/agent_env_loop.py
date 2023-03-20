
import numpy as np

def agent_env_loop(env,agent,args,folder,episodeCount,episodeLimit,using_nextstate=False):

    env.uncertainty_set = agent.uncertainty_set
    env.uncertainty_set.centroids = env.states
    saving_frequency=args.saving_freq
    solved = False  # Whether the solved requirement is met

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount <= episodeLimit:
        s = env.reset()  # Reset robot and get starting observation
        print("start episode at state ",s)
        env.episodeScore = 0
        env.episodeConstraint = 0
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

            env.episodeScore += r  # Accumulate episode reward
            env.episodeConstraint += c  # Accumulate episode constraint
            if done:
                break

            s = s_next  # state for next step is current step's newState

        print("Episode #", episodeCount, "score:", env.episodeScore)
        print("Episode #", episodeCount, "constraint:", env.episodeConstraint)
        # The average action probability tells us how confident the agent was of its actions.
        # By looking at this we can check whether the agent is converging to a certain policy.
        maxActionProb = np.max(actionProbsList)
        #averageEpisodeActionProbs.append(avgActionProb)
        print("Max action prob:", maxActionProb)

        # update the agent
        agent.uncertainty_set.add_visits(agent.buffer)
        agent.uncertainty_set.set_params()
        env.uncertainty_set = agent.uncertainty_set
        env.episodeScoreList.append(env.episodeScore)
        env.episodeConstraintList.append(env.episodeConstraint)
        agent.trainStep(batchSize=step + 1)
        if (episodeCount % saving_frequency) == 0:
            print("saving at episode count " + str(episodeCount))
            agent.save(folder, episodeCount)
        solved = env.solved()  # Check whether the task is solved
        episodeCount += 1  # Increment episode counter

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    #movingAvgN = 10
    #plotData(convolve(env.episodeScoreList, ones((movingAvgN,))/movingAvgN, mode='valid'),
    #         "episode", "episode score", "Episode scores over episodes")
    #plotData(convolve(averageEpisodeActionProbs, ones((movingAvgN,))/movingAvgN, mode='valid'),
    #        "episode", "average episode action probability", "Average episode action probability over episodes")

    if not solved:
        print("Reached episode limit and task was not solved, deploying agent for testing...")
    else:
        print("Task is solved, deploying agent for testing...")

    agent.save(folder,episodeCount)

    s = env.reset()
    env.episodeScore = 0
    while True:
        a, grad, actionProbs = agent.work(s, test=True)
        s_next, r, c, grad_adv, probs_adv, done, info = env.step([a])
        env.episodeScore += r  # Accumulate episode reward
        env.episodeConstraint += c  # Accumulate episode constraint
        s = s_next
        if done:
            print("Reward accumulated =", env.episodeScore)
            env.episodeScore = 0
            s = env.reset()
