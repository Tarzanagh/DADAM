function [w, infos] = cdrmsprop(problem, in_options)
% Corrected Decentralized RMSPROP algorithm.

%%
% set dimensions and samples
d = problem.dim();
n = problem.samples();
sample_node= in_options.sample_node;
num_nod = in_options.number_of_nodes;
%


% set local options
local_options.beta1 = 0.9;
local_options.beta2 = 0.999;
local_options.beta3 = 0.9;
local_options.epsilon = 1e-4;

% merge options
options = mergeOptions(get_default_options(d), local_options);
options = mergeOptions(options, in_options);


% initialize
total_iter = 0;
epoch = 0;
grad_calc_count = 0;
w = options.w_init;
num_of_bachces = floor(n /(num_nod*options.batch_size));
%proximal operator
if ismethod(problem, 'prox')
    prox=1;
else
    prox=0;
end

%
for ii=1:num_nod
    if ii~=num_nod
        start_index = (ii-1) * sample_node  + 1;
        indice_node{ii}  = start_index:start_index+sample_node -1;
    else
        start_index = (ii-1) * sample_node  + 1;
        indice_node{ii} = start_index:n;
    end
end



% store first infos
clear infos;
[infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);

% set start time
start_time = tic();

% display infos
if options.verbose > 0
    fprintf('C-DRMSPROP: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
end


% update step-size
step = options.stepsizefun(total_iter, options);
% increment total iteration numbers
total_iter = total_iter + 1;

w_ex_node=w*ones(1,num_nod);
for ii=1:num_nod
    sam_indice_node{ii} = randsample(indice_node{ii},options.batch_size);
    grad_val_EX(:,ii)= problem.grad(w_ex_node(:,ii),sam_indice_node{ii});
    r_node(:,ii) = (grad_val_EX(:,ii).^2);
    grad_val_EX(:,ii) = grad_val_EX(:,ii) ./ (sqrt(r_node(:,ii)) + options.epsilon);
end
w_new_node=w_ex_node*options.net-step*grad_val_EX;
w=mean(w_new_node,2);

% main loop
while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
    
    for j = 1 : num_of_bachces
        % update step-size
        step = options.stepsizefun(total_iter, options);
        % calculate gradient
        x_old=w_ex_node;
        w_ex_node=w_new_node;
        grad_val_old = grad_val_EX;
        for ii=1:num_nod
            % calculate gradient
            sam_indice_node{ii} = randsample(indice_node{ii},options.batch_size);
            grad_val_EX(:,ii)= problem.grad(w_ex_node(:,ii),sam_indice_node{ii});
            r_node(:,ii) = options.beta1.*r_node(:,ii) + (1 - options.beta1).*(grad_val_EX(:,ii).^2);
            grad_val_EX(:,ii) = grad_val_EX(:,ii) ./ (sqrt(r_node(:,ii)) + options.epsilon);
        end
        w_new_node=w_ex_node*(eye(num_nod)+options.net)-x_old*options.net2-step*(grad_val_EX-grad_val_old);
        %w_new_node=w_ex_node*options.net-step*grad_val_EX;
        %w_new_node=w_ex_node*options.net2-step*grad_val_EX;
        
        %proximal operator
        if prox==1
            w_new_node(:,ii)  = problem.prox(  w_new_node(:,ii) , step);
        end
        
        %
        w=mean(w_new_node,2);
        
        % increment total iteration numbers
        total_iter = total_iter + 1;
        
    end
    
    % measure elapsed time
    elapsed_time = toc(start_time);
    
    % count gradient evaluations
    grad_calc_count = grad_calc_count + j * num_nod*options.batch_size;
    epoch = epoch + 1;
    
    % store infos
    [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);
    
    % display infos
    if options.verbose > 0
        fprintf('C-DRMSPROP: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
    end
end

if optgap < options.tol_optgap
    fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
elseif epoch == options.max_epoch
    fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
end
end

