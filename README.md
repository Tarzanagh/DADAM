# DADAM: A Consensus-Based Distributed Adaptive Gradient Method for Online Optimization

# Introduction

DADAM is a MATLAB package of a collection of decentralized adaptive online and stochastic optimization algorithms. This solves a constrained minimization problem of the form, 
$$\min_{x \in X} \frac{1}{n}\sum_{t=1}^T\sum_{i=1}^n f_{i,t}(x),$$ 
where $f_{i,t}$ is a continuously differentiable function on the closed convex set $X$.  

For more details, please see https://arxiv.org/pdf/1901.09109.pdf

# Installation

 1- Requirement
 
The algorithms have been implemented in MATLAB and make extensive use of the SGDLibrary. You can find the latest version at https://github.com/hiroyuki-kasai/SGDLibrary 


 2- Setup
 
Run run_me_first_to_add_libs_.m for path configurations.

You must then make sure that SGDLibrary-master and DADAM-master can be seen from MATLAB (i.e. make sure to run addpath on their paths).

 3- Simplest usage example
 
Execute example.m for the simplest demonstration of this package. This is the case of softmax regression problem.

# Reproducing experiments from the paper

To reproduce the experiments, execute

dadam_test_linear_svm.m 

dadam_test_l1_logistic_regression.m 

dadam_test_softmax_classifier.m



