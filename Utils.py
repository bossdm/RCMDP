import os

def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("created new folder ", folder)