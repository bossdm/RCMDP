# Robust constrained Markov decision processes (RCMDPs)

This repository implements algorithms for robust constrained Markov decision processes (RCMDPs; [1,2]):

* Robust Constrained Policy Gradient (RCPG) and variants thereof 
* Adversarial RCPG 

and related ablations:

* CPG: ablation without robustness
* PG: ablation without constraints, corresponds to REINFORCE


[1] R. H. Russel, M. Benosman, and J. Van Baar (2021). “Robust Constrained-
MDPs: Soft-Constrained Robust Policy Optimization under Model
Uncertainty.” Advances in Neural Information Processing Systems
workshop (NeurIPS 2021).
https://arxiv.org/abs/2010.04870

[2] D. M. Bossens (2024). "Robust Lagrangian and 
Adversarial Policy Gradient for Robust Constrained Markov Decision Processes." 
IEEE Conference on Artificial Intelligence (CAI 2024).
https://arxiv.org/abs/2308.11267


# Specifications

Tested on python 3.8

Dependencies: Keras and Tensorflow


# Running the algorithm

You can run the algorithm on the experiments from [1] with the following commands.

Run the algorithm on SafeNavigation1:

``python SafeNavigation1.py``

Run the algorithm on SafeNavigation2:

``python SafeNavigation2.py``

Run the algorithm on InventoryManagement:

``python InventoryManagement.py``