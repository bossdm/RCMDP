
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

def get_CVaR(alpha,data,reverse):
    N = int(round(alpha*len(data)))
    sorted_data = sorted(data,reverse=reverse)[0:N]
    F_alpha = sorted_data[-1]
    CVaR = np.mean(sorted_data)
    return F_alpha,CVaR

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
        print(line, r)
        data.append([float(r[c]) for c in columns])
    return data

def plot_sim_overshoot_development(folder,methods,labels,runs,d,its,snaps,tag): # data comes from simcmdp_log.txt
    fig, ax = plt.subplots()
    step = its//snaps
    plots = []
    for method in methods:
        plotline=[]
        for run in runs:
            try:
                data = get_data_from_file(folder+method+"/run"+str(run)+"/simcmdp_log.txt",columns=[1],lines=None,linesExclude=[0])
                data = [dat[0] for dat in data]  # flatten the list of lists
            except:
                file = open("missing.txt","a")
                file.write(method + " " + str(run) + " \n")
                continue

            plotline.append(np.array(data)[0:its:step])
        np.stack(plotline)
        m = np.mean(plotline,axis=0) - d
        s = np.std(plotline,axis=0)/np.sqrt(len(runs))
        x=np.array(list(range(0,its,step)))
        plots.append(ax.plot(x,m)[0])
        ax.fill_between(x, m-s,m+s,alpha=0.25)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Overshoot')
    ax.legend(plots,labels)
    plt.savefig("sim_overshoot_development_"+tag+".pdf")

def plot_sim_value_development(folder,methods,labels,runs,its,snaps,tag): # data comes from simcmdp_log.txt
    fig, ax = plt.subplots()
    step = its // snaps
    plots = []
    for method in methods:
        plotline=[]
        for run in runs:
            try:
                data = get_data_from_file(folder+method+"/run"+str(run)+"/simcmdp_log.txt",columns=[0],lines=None,linesExclude=[0])
                data = [dat[0] for dat in data] # flatten the list of lists
            except:
                file = open("missing.txt", "a")
                file.write(method + " " + str(run) + " \n")
                input(method + str(run))
            plotline.append(np.array(data)[0:its:step])

        np.stack(plotline)
        m = np.mean(plotline,axis=0)
        s = np.std(plotline,axis=0)/np.sqrt(len(runs))
        x=np.array(list(range(0,its,step)))
        plots.append(ax.plot(x,m)[0])
        ax.fill_between(x, m-s,m+s,alpha=0.25)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Value')
    ax.legend(plots,labels)
    plt.savefig("sim_value_development_"+tag+".pdf")

def table_test_overshoot(folder,methods,runs,d,tag,N_perturbs,start,parameter): # data comes from test_performance_stochastic.txt
    file=open("test_overshoot_"+tag+parameter+".txt","w")
    file.write(r"& Mean & $F_{\alpha}$ & $\text{CVaR}_{\alpha}$")
    file.write("\n")
    for i, method in enumerate(methods):
        means = []
        CVaRs = []
        F_alphas = []
        end = start + N_perturbs * test_its
        for run in runs:
            data = get_data_from_file(folder + method + "/run" + str(run) + "/real_cmdp_log.txt", columns=[1], lines=range(start,end))
            data = [dat[0] for dat in data]  # flatten the list of lists
            means.append(np.mean(data))
            F_alpha, CVaR = get_CVaR(alpha=0.10,data=data,reverse=True)
            F_alphas.append(F_alpha)
            CVaRs.append(CVaR)
        np.stack(means)
        overshoots = np.clip(np.array(means) - d,0,float("inf"))
        m = np.mean(overshoots,axis=0)
        s = np.std(means, axis=0)/np.sqrt(len(runs))
        overshoots = np.clip(np.array(F_alphas) - d,0,float("inf"))
        F = np.mean(overshoots,axis=0)
        s_F = np.std(F_alphas, axis=0) / np.sqrt(len(runs))
        overshoots = np.clip(np.array(CVaRs) - d, 0, float("inf"))
        C = np.mean(overshoots,axis=0)
        s_C = np.std(CVaRs, axis=0) / np.sqrt(len(runs))
        file.write(labels[i] + " ")
        file.write(r"& $ %.1f \pm %.1f$ & $%.1f \pm %.1f$ & $%.1f \pm %.1f$ "%(m,s,F,s_F,C,s_C))
        file.write("\n")

def table_test_value(folder,methods,runs,tag,N_perturbs,start,parameter): # data comes from test_performance_stochastic.txt
    file=open("test_value_"+tag+parameter+".txt","w")
    file.write(r"& Mean & $F_{\alpha}$ & $\text{CVaR}_{\alpha}$")
    file.write("\n")
    for i, method in enumerate(methods):
        means = []
        CVaRs = []
        F_alphas = []
        end =  start + N_perturbs * test_its
        for run in runs:
            data = get_data_from_file(folder + method + "/run" + str(run) + "/real_cmdp_log.txt", columns=[0], lines=range(start,end))
            data = [dat[0] for dat in data]  # flatten the list of lists
            means.append(np.mean(data))
            F_alpha, CVaR = get_CVaR(alpha=0.10,data=data,reverse=False)
            F_alphas.append(F_alpha)
            CVaRs.append(CVaR)
        np.stack(means)
        m = np.mean(means,axis=0)
        s = np.std(means, axis=0)/np.sqrt(len(runs))
        F = np.mean(F_alphas,axis=0)
        s_F = np.std(F_alphas, axis=0) / np.sqrt(len(runs))
        C = np.mean(CVaRs,axis=0)
        s_C = np.std(CVaRs, axis=0) / np.sqrt(len(runs))
        file.write(labels[i] + " ")
        file.write(r"& $ %.1f \pm %.1f$ & $%.1f \pm %.1f$ & $%.1f \pm %.1f$ "%(m,s,F,s_F,C,s_C))
        file.write("\n")

def table_test_Rpenalised(folder,methods,runs,tag,scale,N_perturbs,start,parameter):
    file = open("test_Rpenalised_" + tag + parameter + ".txt", "w")
    file.write(r"& $R_{pen}$ (signed) & $R_{pen}$ (positive)")
    file.write("\n")
    for i, method in enumerate(methods):
        values=[]
        overshoots=[]
        absolute_overshoots=[]
        end = start + N_perturbs * test_its
        for run in runs:
            data = get_data_from_file(folder + method + "/run" + str(run) +  "/real_cmdp_log.txt", columns=[0,1],
                                      lines=range(start,end))
            value = np.mean([dat[0] for dat in data])  # flatten the list of lists
            overshoot = np.mean([dat[1] for dat in data]) - d  # flatten the list of lists
            values.append(value)
            overshoots.append(overshoot)
            absolute_overshoots.append(max(0, overshoot))

        Rpen = np.array(values) - scale*np.array(overshoots)
        m = np.mean(Rpen)
        s = np.std(Rpen) / np.sqrt(len(runs))
        Rpen2 = np.array(values) - scale * np.array(absolute_overshoots)
        m2 = np.mean(Rpen2)
        s2 = np.std(Rpen2) / np.sqrt(len(runs))

        file.write(labels[i] + " ")
        file.write(r"& $ %.1f \pm %.1f$ & $ %.1f \pm %.1f$" % (m, s, m2, s2))
        file.write("\n")

def plot_test_overshoot_by_perturbation(folder,methods,labels,runs,d,begin,perturbs,test_its,tag,parameter="Parameter"): # data comes from last N_test*test_its data points in the real_cmdp_log.txt
    fig, ax = plt.subplots()
    N_perturbs=len(perturbs)
    plots = []
    for method in methods:
        ms=[]
        stds=[]
        start=begin
        for perturb in range(N_perturbs):
            plotline = []
            for run in runs:
                data = get_data_from_file(folder + method + "/run" + str(run) + "/real_cmdp_log.txt", columns=[1], lines=range(start,start+test_its),
                                      linesExclude=[0])
                data = [dat[0] for dat in data]  # flatten the list of lists
                plotline.append(np.mean(data))
            m = np.mean(plotline) - d
            s = np.std(plotline)/np.sqrt(len(runs))
            ms.append(m)
            stds.append(s/len(runs))
            start += test_its
        x = perturbs
        ms=np.array(ms)
        stds=np.array(stds)
        plots.append(ax.plot(x, ms)[0])
        ax.fill_between(x, ms - stds, ms + stds,alpha=0.25)
    ax.set_xlabel(parameter)
    ax.set_ylabel('Overshoot')
    ax.legend(plots,labels)
    plt.savefig("test_overshoot_by_perturbation_"+tag+parameter+".pdf")

def plot_test_value_by_perturbation(folder,methods,labels,runs,perturbs,test_its,begin,tag,parameter="Parameter"): # data comes from last N_test*test_its data points in the real_cmdp_log.txt
    fig, ax = plt.subplots()
    N_perturbs=len(perturbs)
    plots=[]
    for method in methods:
        ms=[]
        stds=[]
        start = begin
        for perturb in range(N_perturbs):
            plotline = []
            for run in runs:
                data = get_data_from_file(folder + method + "/run" + str(run) + "/real_cmdp_log.txt", columns=[0], lines=range(start,start+test_its),
                                      linesExclude=[0])
                data = [dat[0] for dat in data]  # flatten the list of lists
                plotline.append(np.mean(data))
            m = np.mean(plotline)
            s = np.std(plotline)/np.sqrt(len(runs))
            ms.append(m)
            stds.append(s)
            start += test_its
        x = perturbs
        ms=np.array(ms)
        stds=np.array(stds)
        plots.append(ax.plot(x, ms)[0])
        ax.fill_between(x, ms - stds, ms + stds,alpha=0.25)
    ax.set_xlabel(parameter)
    ax.set_ylabel('Value')
    ax.legend(plots,labels)
    plt.savefig("test_value_by_perturbation_"+tag+parameter+".pdf")

if __name__ == "__main__":
    labels = ["Adversarial RCPG","RCPG (Robust Value)", "RCPG (Robust Constraint)","CPG", "PG"]
    runs=range(1,31)
    perturbs=[0.6,0.7,0.8,0.9,1.0]
    test_its=50
    sim_its=5000
    snaps=20
    gamma = 0.99
    H=gamma/(1-gamma)
    factor = H / 200.
    d=4 / factor
    tag="FINAL"
    folder="ResultsFinal100CI90/"
    methods=["AdversarialRCPG_Hoeffding","RCPG_Hoeffding_V","RCPG_Hoeffding_C","CPG","PG"]#"RCPG_Hoeffding_L","CPG","PG"]
    start=-1250
    plot_sim_overshoot_development(folder=folder,methods=methods,labels=labels,runs=runs,d=d,its=sim_its,snaps=20,tag=tag)
    plot_sim_value_development(folder=folder, methods=methods,labels=labels, runs=runs,its=sim_its,snaps=20,tag=tag)
    table_test_value(folder=folder,methods=methods,runs=runs,tag=tag,N_perturbs=len(perturbs),start = start, parameter="Psuccess")  # take the stochastic runs rather than deterministic runs)
    table_test_overshoot(folder=folder, methods=methods, runs=runs,d=d,tag=tag,N_perturbs=len(perturbs),start = start, parameter="Psuccess")
    # test Psuccess
    perturbs = [0.6,0.7,0.8,0.9,1.0]
    plot_test_value_by_perturbation(folder=folder,methods=methods,labels=labels,runs=runs,perturbs=perturbs, begin=start,test_its=test_its,
                                    tag=tag,parameter=r"$P_{success}$")
    plot_test_overshoot_by_perturbation(folder=folder, methods=methods, labels=labels,runs=runs,d=d,
                                    perturbs=perturbs, begin=start,test_its=test_its,tag=tag,parameter=r"$P_{success}$")
    # V_max = 200
    # C_max = 200
    scale = 500 # just take the maximal lagrangian multiplier
    table_test_Rpenalised(folder, methods, runs, tag, scale, len(perturbs),start=start,parameter="Psuccess")

    # test delta
    perturbs = [0.05,0.10,0.25,0.50,1.0] # 100 state-action pairs (25 * 4), and N in {5,10,25,50,100}
    test_its = 50
    start = -500

    plot_test_value_by_perturbation(folder=folder, methods=methods, labels=labels, runs=runs, perturbs=perturbs,begin=start,
                                    test_its=test_its,
                                    tag=tag, parameter=r"$P_{\epsilon}$")
    plot_test_overshoot_by_perturbation(folder=folder, methods=methods, labels=labels, runs=runs, d=d,
                                         perturbs=perturbs, begin=start, test_its=test_its, tag=tag, parameter=r"$P_{\epsilon}$")
    table_test_value(folder=folder,methods=methods,runs=runs,tag=tag,N_perturbs=len(perturbs),start=start,parameter="Pepsilon")
    table_test_overshoot(folder=folder, methods=methods, runs=runs,d=d,tag=tag,N_perturbs=len(perturbs),start=start,parameter="Pepsilon")
    # V_max = 200
    # C_max = 200
    scale = 500 #
    table_test_Rpenalised(folder, methods, runs, tag, scale, len(perturbs),start=start,parameter="Pepsilon")
