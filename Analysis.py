
import sys, os

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))
from RCMDP.Choose_Method import *
import argparse
import numpy as np

parser = argparse.ArgumentParser(
                    prog = 'Analysis',
                    description = 'makes plots and tables')
parser.add_argument('--folder',dest="folder",type=str,default="")
args = parser.parse_args()

def get_data_from_file(filename,columns,lines=None,linesExclude=None):
    rows=open(filename,'r').read().split('\n')
    rows = [row for row in rows if row.strip()]
    data=[]
    if lines is None:
        lines=range(len(rows))
        if linesExclude is not None:
            lines = [line for line in lines if line not in linesExclude]

    for line in lines:
        row = rows[line]
        r = row.strip().split('\t')
        print(r)
        data.append([float(r[c]) for c in columns])
    return data

def plot_sim_overshoot_development(folder,methods,runs,d): # data comes from simcmdp_log.txt
    fig, ax = plt.subplots()
    for method in methods:
        plotline=[]
        for run in runs:
            data = get_data_from_file(folder+method+"/run"+str(run)+"/simcmdp_log.txt",columns=[1],lines=None,linesExclude=[0])
            data = [dat[0] for dat in data] # flatten the list of lists
            plotline.append(np.array(data))
        np.stack(plotline)
        m = np.mean(plotline,axis=0) - d
        s = np.std(plotline,axis=0)
        x=np.array(list(range(len(plotline[0]))))
        ax.plot(x,m)
        ax.fill_between(x, m-s,m+s)
    plt.savefig("sim_overshoot_development.pdf")

def plot_sim_value_development(folder,methods,runs): # data comes from simcmdp_log.txt
    fig, ax = plt.subplots()
    for method in methods:
        plotline=[]
        for run in runs:
            data = get_data_from_file(folder+method+"/run"+str(run)+"/simcmdp_log.txt",columns=[0],lines=None,linesExclude=[0])
            data = [dat[0] for dat in data] # flatten the list of lists
            plotline.append(np.array(data))
        np.stack(plotline)
        m = np.mean(plotline,axis=0)
        s = np.std(plotline,axis=0)
        x=np.array(list(range(len(plotline[0]))))
        ax.plot(x,m)
        ax.fill_between(x, m-s,m+s)
    plt.savefig("sim_value_development.pdf")

def table_test_overshoot(folder,methods,runs,d): # data comes from test_performance_stochastic.txt
    file=open("test_overshoot.txt","w")
    for method in methods:
        plotline = []
        for run in runs:
            data = get_data_from_file(folder + method + "/run" + str(run) + "/test_performance_stochastic.txt", columns=[2], lines=None)
            data = [dat[0] for dat in data]  # flatten the list of lists
            plotline.append(np.array(data))
        np.stack(plotline)
        m = np.mean(plotline, axis=0) - d
        s = np.std(plotline, axis=0)
        file.write(r"& $ %.4f \pm %.2f $ "%(m,s))

def table_test_value(folder,methods,runs): # data comes from test_performance_stochastic.txt
    file=open("test_value.txt","w")
    for method in methods:
        plotline = []
        for run in runs:
            data = get_data_from_file(folder + method + "/run" + str(run) + "/test_performance_stochastic.txt", columns=[0], lines=None)
            data = [dat[0] for dat in data]  # flatten the list of lists
            plotline.append(np.array(data))
        np.stack(plotline)
        m = np.mean(plotline, axis=0)
        s = np.std(plotline, axis=0)
        file.write(r"& $ %.4f \pm %.2f $ "%(m,s))

def plot_test_overshoot_by_perturbation(folder,methods,runs,d,perturbs,test_its): # data comes from last N_test*test_its data points in the real_cmdp_log.txt
    fig, ax = plt.subplots()
    N_perturbs=len(perturbs)
    for method in methods:
        ms=[]
        stds=[]
        start = -N_perturbs*test_its
        for perturb in range(N_perturbs):
            plotline = []
            for run in runs:
                data = get_data_from_file(folder + method + "/run" + str(run) + "/real_cmdp_log.txt", columns=[1], lines=range(start,start+test_its),
                                      linesExclude=[0])
                data = [dat[0] for dat in data]  # flatten the list of lists
                plotline.append(np.mean(data))
            m = np.mean(plotline) - d
            s = np.std(plotline)
            ms.append(m)
            stds.append(s)
            start += test_its
        x = perturbs
        ms=np.array(ms)
        stds=np.array(stds)
        ax.plot(x, ms)
        ax.fill_between(x, ms - stds, ms + stds)
    plt.savefig("test_overshoot_by_perturbation.pdf")

def plot_test_value_by_perturbation(folder,methods,runs,perturbs,test_its): # data comes from last N_test*test_its data points in the real_cmdp_log.txt
    fig, ax = plt.subplots()
    N_perturbs=len(perturbs)
    for method in methods:
        ms=[]
        stds=[]
        start = -N_perturbs*test_its
        for perturb in range(N_perturbs):
            plotline = []
            for run in runs:
                data = get_data_from_file(folder + method + "/run" + str(run) + "/real_cmdp_log.txt", columns=[0], lines=range(start,start+test_its),
                                      linesExclude=[0])
                data = [dat[0] for dat in data]  # flatten the list of lists
                plotline.append(np.mean(data))
            m = np.mean(plotline)
            s = np.std(plotline)
            ms.append(m)
            stds.append(s)
            start += test_its
        x = perturbs
        ms=np.array(ms)
        stds=np.array(stds)
        ax.plot(x, ms)
        ax.fill_between(x, ms - stds, ms + stds)
    plt.savefig("test_value_by_perturbation.pdf")

if __name__ == "__main__":
    runs=range(1)
    perturbs=[0.6,0.7,0.8,0.9,1.0]
    test_its=10
    d=4
    folder=args.folder
    methods=["AdversarialRCPG_Hoeffding"] # ["AdversarialRCPG_Hoeffding","RCPG_Hoeffding","CPG","PG"]
    plot_test_value_by_perturbation(folder=folder,methods=methods,runs=runs,perturbs=perturbs, test_its=test_its)
    plot_test_overshoot_by_perturbation(folder=folder, methods=methods, runs=runs,d=d,
                                    perturbs=perturbs, test_its=test_its)
    plot_sim_overshoot_development(folder=folder,methods=methods,runs=runs,d=d)
    plot_sim_value_development(folder=folder, methods=methods, runs=runs)
    table_test_value(folder=folder,methods=methods,runs=runs)
    table_test_overshoot(folder=folder, methods=methods, runs=runs,d=d)