function  dadam_test_linear_svm()

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

% Created on Jan. 17, 2019
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
if 1     % generate synthetic data
    n = 1000;    % # of samples per class
    d = 100;      % # of dimensions
    std = 0.15; % standard deviation
    
    data = multiclass_data_generator(n, d, l, std);
    d = d + 1; % adding '1' row for intersect
    
    % train data
    x_train = [data.x_train; ones(1,l*n)];
    % assign y (label) {1,-1}
    y_train(data.y_train<=1.5) = -1;
    y_train(data.y_train>1.5) = 1;
    
    % test data
    x_test = [data.x_test; ones(1,l*n)];
    % assign y (label) {1,-1}
    y_test(data.y_test<=1.5) = -1;
    y_test(data.y_test>1.5) = 1;
    
else    % load real-world data
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
    
end
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

%% plot all
% display cost vs grads
display_graph('grad_calc_count','cost', algorithms, w_list, info_list);
% display_graph('time','cost', algorithms, w_list, info_list);
%
% % display optimality gap vs grads
% if options.f_opt ~= -Inf
%     display_graph('grad_calc_count','optimality_gap', algorithms, w_list, info_list);
% end

% display classification results
y_pred_list = cell(length(algorithms),1);
accuracy_list = cell(length(algorithms),1);
for alg_idx=1:length(algorithms)
    if ~isempty(w_list{alg_idx})
        p = problem.prediction(w_list{alg_idx});
        % calculate accuracy
        accuracy_list{alg_idx} = problem.accuracy(p);
        
        fprintf('Classificaiton accuracy: %s: %.4f\n', algorithms{alg_idx}, problem.accuracy(p));
        
        % convert from {1,-1} to {1,2}
        p(p==-1) = 2;
        p(p==1) = 1;
        % predict class
        y_pred_list{alg_idx} = p;
    else
        fprintf('Classificaiton accuracy: %s: Not supported\n', algorithms{alg_idx});
    end
end

% convert from {1,-1} to {1,2}
y_train(y_train==-1) = 2;
y_train(y_train==1) = 1;
y_test(y_test==-1) = 2;
y_test(y_test==1) = 1;
if plot_flag
    display_classification_result(problem, algorithms, w_list, y_pred_list, accuracy_list, x_train, y_train, x_test, y_test);
end

end


