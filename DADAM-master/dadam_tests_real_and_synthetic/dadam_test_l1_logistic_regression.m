function dadam_test_l1_logistic_regression()

%% The problem of interest is defined as
%
%           min f(w) = 1/n * sum_i^n f_i(w),
%           where
%           f_i(w) = log(1 + exp(-y_i' .* (w'*x_i))) + lambda || w ||_1.
%
% "w" is the model parameter of size d vector.
%
%
% This file is part of DADAM-master.

% Created by D.Ataee Tarzanagh on Jan. 17, 2019
% Note that partial code is originaly created by H.Kasai for centralized algorithms (See https://github.com/hiroyuki-kasai/SGDLibrary)


%rng('default');

clc;
clear;
close all;

%% Set algorithms
algorithms={
    'SGD','ADAGRAD','ADADELTA','RMSPROP','ADAM',...
    'DSGD','DADAGRAD','DADADELTA','DRMSPROP','DADAM',...
    'C-DSGD','C-DADAGRAD','C-DADADELTA','C-DRMSPROP','C-DADAM',...
    };

%% prepare dataset
if 0
    % generate synthtic data
    d = 100;
    n = 1000;
    data = logistic_regression_data_generator(n, d);
    x_train = data.x_train;
    y_train = data.y_train;
    x_test = data.x_test;
    y_test = data.y_test;
    d = size(x_train,1);
    %w_opt = data.w_opt;
    lambda = 0.1;
else
    % load pre-created synthetic data
    data = importdata('../data/logistic_regression/data_100d_10000.mat');
    x_train = data.x_train;
    y_train = data.y_train;
    x_test = data.x_test;
    y_test = data.y_test;
    d = size(x_train,1);
    n = length(y_train);
    w_opt = data.w_star;
    lambda =data.lambda;
end

%% define problem definitions
problem = l1_logistic_regression(x_train, y_train, x_test, y_test, lambda);


%% initialize
w_init = rand(d,1);
batch_size = 10;
w_list = cell(length(algorithms),1);
info_list = cell(length(algorithms),1);


%% calculate solution
%if norm(w_opt)
%else
% calculate solution
options.w_init = w_init;
options.max_iter = 200;
w_opt = problem.calc_solution(options, 'sd_nesterov');
%end
f_opt = problem.cost(w_opt);
fprintf('f_opt: %.24e\n', f_opt);


clear options;
%% general options for optimization algorithms
options.w_init = w_init;
options.tol = 10^-24;
options.max_epoch = 100;
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

%% perform algorithms
for alg_idx=1:length(algorithms)
    fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
    
    
    
    switch algorithms{alg_idx}
        
        case {'DSGD'}
            
            [w_list{alg_idx}, info_list{alg_idx}] = dsgd(problem, options);
            
            % AdaGrad variants
        case {'DADAGRAD'}
            
            options.epsilon = 1e-4;
            [w_list{alg_idx}, info_list{alg_idx}] = dadagrad(problem, options);
            
        case {'DRMSPROP'}
            options.epsilon = 1e-4;
            [w_list{alg_idx}, info_list{alg_idx}] = drmsprop(problem, options);
            
        case {'DADADELTA'}
            
            options.epsilon = 1e-4;
            [w_list{alg_idx}, info_list{alg_idx}] = dadadelta(problem, options);
            
        case {'DADAM'}
            
            options.epsilon = 1e-4;
            [w_list{alg_idx}, info_list{alg_idx}] = dadam(problem, options);
        case {'DADAMAX'}
            
            options.epsilon = 1e-4;
            
            [w_list{alg_idx}, info_list{alg_idx}] = dadamax(problem, options);
            
        case {'C-DSGD'}
            
            [w_list{alg_idx}, info_list{alg_idx}] = cdsgd(problem, options);
            
            % AdaGrad variants
        case {'C-DADAGRAD'}
            
            options.epsilon = 1e-4;
            [w_list{alg_idx}, info_list{alg_idx}] = cdadagrad(problem, options);
            
        case {'C-DRMSPROP'}
            options.epsilon = 1e-4;
            [w_list{alg_idx}, info_list{alg_idx}] = cdrmsprop(problem, options);
            
        case {'C-DADADELTA'}
            
            options.epsilon = 1e-4;
            [w_list{alg_idx}, info_list{alg_idx}] = cdadadelta(problem, options);
            
        case {'C-DADAM'}
            
            options.epsilon = 1e-4;
            
            [w_list{alg_idx}, info_list{alg_idx}] = cdadam(problem, options);
            
        case {'SGD'}
            
            options.epsilon = 1e-4;
            options.step_init = 0.001 * options.batch_size; %
            
            
            [w_list{alg_idx}, info_list{alg_idx}] = sgd(problem, options);
            
            
            % AdaGrad variants
        case {'ADAGRAD'}
            
            options.epsilon = 1e-4;
            options.sub_mode = 'AdaGrad';
            options.step_init = 0.001 * options.batch_size; %
            
            [w_list{alg_idx}, info_list{alg_idx}] = adagrad(problem, options);
            
        case {'RMSPROP'}
            
            options.epsilon = 1e-4;
            options.sub_mode = 'RMSProp';
            options.step_init = 0.001 * options.batch_size; %
            
            
            [w_list{alg_idx}, info_list{alg_idx}] = adagrad(problem, options);
            
        case {'ADADELTA'}
            
            options.epsilon = 1e-4;
            options.sub_mode = 'AdaDelta';
            options.step_init = 0.001 * options.batch_size; %
            
            [w_list{alg_idx}, info_list{alg_idx}] = adagrad(problem, options);
            
        case {'ADAM'}
            
            options.epsilon = 1e-4;
            options.sub_mode = 'Adam';
            options.step_init = 0.001 * options.batch_size; %
            
            
            [w_list{alg_idx}, info_list{alg_idx}] = adam(problem, options);
            
        otherwise
            warn_str = [algorithms{alg_idx}, ' is not supported.'];
            warning(warn_str);
            w_list{alg_idx} = '';
            info_list{alg_idx} = '';
    end
    
end



fprintf('\n\n');


%% plot all

display_graph('grad_calc_count','cost', algorithms, w_list, info_list);

end




