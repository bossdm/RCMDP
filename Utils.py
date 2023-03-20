import os

def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("created new folder ", folder)

def resume(folder,agent):
    if not os.path.exists(folder):
        print("starting training from scratch; no folder " + folder + " found ")
        return 1 # episode count
    else:
        last_episode = 0
        paths = os.listdir(folder)
        if paths:
            # print(paths)
            for path in paths:
                episode = int(path[7:])
                # print("episode ",episode)
                if episode > last_episode:
                    last_episode = episode
            agent.load(folder, last_episode)
            print("continuing from checkpoint at episode ", last_episode)
            return last_episode + 1
        else:
            print("starting training from scratch; folder " + folder + " is empty ")
            return 1