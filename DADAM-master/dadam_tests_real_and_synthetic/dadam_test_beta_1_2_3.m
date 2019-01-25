function dadam_test_beta_1_2_3()
%% Sensitivity of DADAM to  decay parameters

% The problem of interest is defined as
%
%           min f(w) = 1/n * sum_i^n f_i(w),
%           where
%           f_i(w) = 1/2 * (max(0.0, 1 - y_i .* (w'*x_i) )^2 + lambda/2 * w^2.
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of DADAM.

% Created by D.Ataee Tarzanagh on Jan. 17, 2019
% Note that partial code is originaly created by H.Kasai for centralized algorithms (See https://github.com/hiroyuki-kasai/SGDLibrary)
clc;
clear;
close all;

%% Set algorithms
algorithms={
    'SGD','ADAGRAD','ADADELTA','RMSPROP','ADAM',...
    'DSGD','DADAGRAD','DADADELTA','DRMSPROP','DADAM',...
    'C-DSGD','C-DADAGRAD','C-DADADELTA','C-DRMSPROP','C-DADAM',...
    };


%% # of classes (must not change)
l = 2;

%% prepare dataset
% load real-world data
data = importdata('../data/mushroom/mushroom.mat');
n = size(data.X,1);
d = size(data.X,2) + 1;
x_in = [data.X ones(n,1)]';
y_in = data.y';

perm_idx = randperm(n);
x = x_in(:,perm_idx);
y = y_in(perm_idx);

% split data into train and test data
% train data
n_train = floor(n/8);
x_train = x(:,1:n_train);
y_train = y(1:n_train);
x_train_class1 = x_train(:,y_train>0);
x_train_class2 = x_train(:,y_train<0);
n_class1 = size(x_train_class1,2);
n_class2 = size(x_train_class2,2);

% test data
x_test = x(:,n_train+1:end);
y_test = y(n_train+1:end);
x_test_class1 = x_test(:,y_test>0);
x_test_class2 = x_test(:,y_test<0);
n_test_class1 = size(x_test_class1,2);
n_test_class2 = size(x_test_class2,2);
n_test = n_test_class1 + n_test_class2;

lambda = 0.1;
w_opt = zeros(d,1);

% set plot_flag
if d > 4
    plot_flag = false;  % too high dimension
else
    plot_flag = true;
end


%% define problem definitions
problem = linear_svm(x_train, y_train, x_test, y_test, lambda);


%% initialize
w_init = randn(d,1);
batch_size = 10;
w_list = cell(length(algorithms),1);
info_list = cell(length(algorithms),1);


%% calculate solution
if norm(w_opt)
else
    % calculate solution
    w_opt = problem.calc_solution(1000);
end
f_opt = problem.cost(w_opt);
fprintf('f_opt: %.24e\n', f_opt);



clear options;
%% general options for optimization algorithms
options.w_init = w_init;
options.tol = 10^-24;
options.max_epoch = 500;
options.verbose = true;
options.lambda = lambda;
options.permute_on = 1;
options.f_opt = f_opt;

%% data distribution
options.batch_size = batch_size;
options.number_of_nodes = 10;
options.sample_node= floor(problem.samples/(options.number_of_nodes*options.batch_size));

%% define network -- random graph options

quiet=0; show_graph=0; per=0.5;eps_deg=1;
[~,P,~,~]=random_graph_producer2(options.number_of_nodes,per,'random',show_graph,quiet);
W_net=local_degree(P,options.number_of_nodes,eps_deg);
options.net = W_net;
options.net2=(eye(options.number_of_nodes)+W_net)/2;
options.step_alg = 'decay-sq'; % or 'fix',
eig_W=sort(eig(options.net));
options.step_init = sqrt(1-eig_W(2)); %

%%
options.beta1 = 0.9;
options.beta2 = 0.999;
options.beta3 = 0.00; %'AMSGRAD'
[w_sgd_fix, info_sgd_fix] = dadam(problem, options);


options.beta1 = 0.9;
options.beta2 = 0.999;
options.beta3 = 0.9;  %DADAM
[w_sgd_decay2, info_sgd_decay2] = dadam(problem, options);


options.beta1 = 0.9;
options.beta2 = 0.999;
options.beta3 = 1; %'ADAM'

[w_sgd_my, info_sgd_my] = dadam(problem, options);


%% display cost/optimality gap vs number of gradient evaluations
display_graph('grad_calc_count','cost', {'AMSGRAD (\beta_3=0)','DADAM (\beta_3=0.9)', 'ADAM (\beta_3=1)'}, ...
    {w_sgd_fix, w_sgd_decay2, w_sgd_my}, {info_sgd_fix,  info_sgd_decay2, info_sgd_my});
xlim([1 6000])




