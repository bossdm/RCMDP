
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
    print(filename)
    rows = [row for row in rows if row.strip()]
    data=[]
    if lines is None:
        row_len = len(rows)
        lines=range(row_len)
        if linesExclude is not None:
            lines = [line for line in lines if line not in linesExclude]

    for line in lines:
        print(line)
        row = rows[line]
        r = row.strip().split('\t')
        print(r)
        data.append([float(r[c]) for c in columns])
    return data

def plot_sim_overshoot_development(folder,methods,labels,markers,runs,d,its,snaps,tag): # data comes from simcmdp_log.txt
    fig, ax = plt.subplots(figsize=(10,10))
    step = its//snaps
    plots = []
    for i,method in enumerate(methods):
        plotline=[]
        for run in runs:
            data = get_data_from_file(folder+method+"/run"+str(run)+"/performance/train.txt",columns=[1],lines=None,linesExclude=None)
            data = [dat[0] for dat in data] # flatten the list of lists
            plotline.append(np.array(data[0:its:step] + [data[-1]]))
        print(method)
        np.stack(plotline)
        m = np.mean(plotline,axis=0) - d
        s = np.std(plotline,axis=0)/np.sqrt(len(runs))
        x=np.array(list(range(0,its+step,step)))
        plots.append(ax.plot(x,m,marker=markers[i],linewidth=3.0,markersize=20)[0])
        ax.fill_between(x, m-s,m+s,alpha=0.25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Episodes',fontsize=30)
    ax.set_ylabel('Overshoot',fontsize=30)
    plt.tight_layout()
    #ax.legend(plots,labels,loc='upper right')
    plt.savefig("sim_overshoot_development_"+tag+".pdf")

def plot_sim_value_development(folder,methods,labels,markers,runs,its,snaps,tag): # data comes from simcmdp_log.txt
    fig, ax = plt.subplots(figsize=(10,10))
    step = its // snaps
    plots = []
    for i, method in enumerate(methods):
        plotline=[]
        for run in runs:
            data = get_data_from_file(folder+method+"/run"+str(run)+"/performance/train.txt",columns=[0],lines=None,linesExclude=None)
            data = [dat[0] for dat in data] # flatten the list of lists
            plotline.append(np.array(data[0:its:step] + [data[-1]]))
        np.stack(plotline)
        m = np.mean(plotline,axis=0)
        s = np.std(plotline,axis=0)/np.sqrt(len(runs))
        x=np.array(list(range(0,its+step,step)))
        plots.append(ax.plot(x,m,marker=markers[i],linewidth=3.0,markersize=20)[0])
        ax.fill_between(x, m-s,m+s,alpha=0.25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Episodes',fontsize=30)
    ax.set_ylabel('Value',fontsize=30)
    plt.tight_layout()
    #ax.legend(plots,labels,loc='upper right')
    plt.savefig("sim_value_development_"+tag+".pdf")

def table_test_overshoot(folder,methods,runs,d,tag,stochstring): # data comes from test_performance_stochastic.txt
    file=open("test_overshoot_"+tag+".txt","w")
    file.write(r"& Mean & $F_{\alpha}$ & $\text{CVaR}_{\alpha}$")
    file.write("\n")
    for i, method in enumerate(methods):
        means = []
        CVaRs = []
        F_alphas = []
        for run in runs:
            data = get_data_from_file(folder + method + "/run" + str(run) + "/performance/test"+stochstring+".txt", columns=[1], lines=None)
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
def table_test_value(folder,methods,runs,tag,stochstring): # data comes from test_performance_stochastic.txt
    file=open("test_value_"+tag+".txt","w")
    file.write(r"& Mean & $F_{\alpha}$ & $\text{CVaR}_{\alpha}$")
    file.write("\n")
    for i, method in enumerate(methods):
        means = []
        CVaRs = []
        F_alphas = []
        for run in runs:
            data = get_data_from_file(folder + method + "/run" + str(run) + "/performance/test"+stochstring+".txt", columns=[0], lines=None)
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
def table_test_Rpenalised(folder,methods,runs,tag,scale,stochstring):
    file = open("test_Rpenalised_" + tag + ".txt", "w")
    file.write(r"& $R_{pen}$ (signed) & $R_{pen}$ (positive)")
    file.write("\n")
    for i, method in enumerate(methods):
        file.write(labels[i] + " ")
    file.write("\n")
    for i, method in enumerate(methods):
        values=[]
        overshoots=[]
        absolute_overshoots = []
        for run in runs:
            data = get_data_from_file(folder + method + "/run" + str(run) + "/performance/test"+stochstring+".txt", columns=[0,1],
                                      lines=None)
            value = np.mean([dat[0] for dat in data])  # flatten the list of lists
            overshoot = np.mean([dat[1] for dat in data]) - d  # flatten the list of lists
            values.append(value)
            overshoots.append(overshoot)
            absolute_overshoots.append(max(0,overshoot))
        Rpen = np.array(values) - scale*np.array(overshoots)
        m = np.mean(Rpen)
        s = np.std(Rpen) / np.sqrt(len(runs))
        Rpen2 = np.array(values) - scale * np.array(absolute_overshoots)
        m2 = np.mean(Rpen2)
        s2 = np.std(Rpen2) / np.sqrt(len(runs))
        file.write(r"& $ %.1f \pm %.1f$ & $ %.1f \pm %.1f$ & " % (m,s,m2,s2))
    file.write("\n")


def plot_test_overshoot_by_perturbation(folder,methods,labels, markers,runs,d,perturbs,test_its,tag,stochstring): # data comes from last N_test*test_its data points in the real_cmdp_log.txt
    fig, ax = plt.subplots(figsize=(10,10))
    N_perturbs=len(perturbs)
    # for tests manipulate μ ∈ {S/6, S/4, S/3} and σ ∈ {S/8, S/6, S/4}
    xticklabels = ["("+mu_factor+","+sigma_factor+")" for mu_factor in [r"$\mu=S/6$",r"$\mu=S/4$",r"$\mu=S/3$"]
        for sigma_factor in [r"$\sigma=S/8$",r"$\sigma=S/6$",r"$\sigma=S/4$"]]
    plots = []
    for i,method in enumerate(methods):
        ms=[]
        stds=[]
        start = -N_perturbs*test_its
        for perturb in range(N_perturbs):
            plotline = []
            for run in runs:
                data = get_data_from_file(folder + method + "/run" + str(run) + "/performance/test"+stochstring+".txt", columns=[1], lines=range(start,start+test_its),
                                      linesExclude=None)
                data = [dat[0] for dat in data]  # flatten the list of lists
                plotline.append(np.mean(data))
            m = np.mean(plotline) - d
            s = np.std(plotline)/np.sqrt(len(runs))
            ms.append(m)
            stds.append(s)
            start += test_its
        x = perturbs
        ms=np.array(ms)
        stds=np.array(stds)
        plots.append(ax.plot(x, ms,marker=markers[i],linewidth=3.0,markersize=20)[0])
        ax.fill_between(x, ms - stds, ms + stds,alpha=0.25)
    plt.xticks(list(range(N_perturbs)),xticklabels,
           rotation=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Demand distribution',fontsize=30)
    ax.set_ylabel('Overshoot',fontsize=30)
    plt.tight_layout()
    #ax.legend(plots,labels)
    plt.savefig("test_overshoot_by_perturbation_"+tag+".pdf")

def plot_test_value_by_perturbation(folder,methods,labels, markers,runs,perturbs,test_its,tag,stochstring): # data comes from last N_test*test_its data points in the real_cmdp_log.txt
    fig, ax = plt.subplots(figsize=(10,10))
    N_perturbs=len(perturbs)
    xticklabels = [mu_factor+","+sigma_factor for mu_factor in [r"$\mu=S/6$",r"$\mu=S/4$",r"$\mu=S/3$"]
        for sigma_factor in [r"$\sigma=S/8$",r"$\sigma=S/6$",r"$\sigma=S/4$"]]
    plots=[]
    for i,method in enumerate(methods):
        ms=[]
        stds=[]
        start = -N_perturbs*test_its
        for perturb in range(N_perturbs):
            plotline = []
            for run in runs:
                data = get_data_from_file(folder + method + "/run" + str(run) + "/performance/test"+stochstring+".txt", columns=[0], lines=range(start,start+test_its),
                                      linesExclude=None)
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
        plots.append(ax.plot(x, ms,label=labels[i],marker=markers[i],linewidth=3.0,markersize=20)[0])
        ax.fill_between(x, ms - stds, ms + stds,alpha=0.25)
    plt.xticks(list(range(N_perturbs)),xticklabels,
           rotation=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Demand distribution',fontsize=30)
    ax.set_ylabel('Value',fontsize=30)
    plt.tight_layout()
    plt.savefig("test_value_by_perturbation_"+tag+".pdf")

    figsize = (20,5.0)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center',ncol=len(labels)//2,fontsize=32)
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.savefig('legend.pdf')


if __name__ == "__main__":
    labels = ["Adversarial RCPG","RCPG (Robust Lagrangian)","RCPG (Robust value)","RCPG (Robust constraint)","CPG","PG"] # "RCPG (Robust Lagrangian)",
    markers=["D","^","o","x","+","s"]
    runs=range(1,21) #+ range(12,21)
    perturbs=[1,2,3,4,5,6,7,8,9]
    test_its=50
    sim_its=5000
    snaps=20
    gamma=0.99
    H = 100
    factor = H/sum([gamma**i for i in range(H)])
    d=6.0*factor
    tag="IM"
    folder="IM_Results/"
    methods=["AdversarialRCPG_Hoeffding","RCPG_Hoeffding_L","RCPG_Hoeffding_V","RCPG_Hoeffding_C","CPG","PG"] # "RCPG_Hoeffding_L",

    plot_sim_overshoot_development(folder=folder,methods=methods,labels=labels,markers=markers,runs=runs,d=d,its=sim_its,snaps=20,tag=tag)
    plot_sim_value_development(folder=folder, methods=methods,labels=labels,markers=markers, runs=runs,its=sim_its,snaps=20,tag=tag)

    for stochstr in ["_stoch","_determ"]:
        ttag=tag+stochstr
        table_test_value(folder=folder,methods=methods,runs=runs,tag=ttag,stochstring=stochstr)
        table_test_overshoot(folder=folder, methods=methods, runs=runs,d=d,tag=ttag,stochstring=stochstr)
        plt.rcParams.update({'font.size': 18})
        plot_test_value_by_perturbation(folder=folder,methods=methods,labels=labels,markers=markers,runs=runs,perturbs=perturbs, test_its=test_its,
                                        tag=ttag,stochstring=stochstr)
        plot_test_overshoot_by_perturbation(folder=folder, methods=methods, labels=labels,markers=markers,runs=runs,d=d,
                                         perturbs=perturbs, test_its=test_its,tag=ttag,stochstring=stochstr)
        table_test_Rpenalised(folder, methods, runs, ttag, scale=500,stochstring=stochstr)  # just take the maximum lagrangian multiplier
