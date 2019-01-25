function [w, infos] = dsgd_parallel(problem, in_options)
% Decentralized SGD algorithm.


%%
% set dimensions and samples
d = problem.dim();
n = problem.samples();
sample_node= in_options.sample_node;
num_nod = in_options.number_of_nodes;

% set local options
local_options.beta = 0.9;
local_options.epsilon = 1e-4;

% merge options
options = mergeOptions(get_default_options(d), local_options);
options = mergeOptions(options, in_options);

% initialize
total_iter = 0;
epoch = 0;
grad_calc_count = 0;
num_of_bachces = floor(n /(num_nod*options.batch_size));

w = options.w_init;
%
for ii=1:num_nod
    if ii~=num_nod
        start_index = (ii-1) * sample_node  + 1;
        indice_node{ii}  = start_index:start_index+sample_node -1;
    else
        start_index = (ii-1) * sample_node  + 1;
        indice_node{ii} = start_index:n;
    end
    w_consen_node{ii}=w;
    w_node{ii}=w;
end

%proximal operator
if ismethod(problem, 'prox')
    prox=1;
else
    prox=0;
end

% store first infos
clear infos;
[infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);

% set start time
start_time = tic();

% display infos
if options.verbose > 0
    fprintf('DSGD: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
end

% main loop
while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
    
    
    for j = 1 : num_of_bachces
        
        % update step-size
        step = options.stepsizefun(total_iter, options)*alpha;
        
        
        parfor ii = 1 : num_nod % can be run in parallel
            
            % calculate gradient
            sam_indice_node{ii} = randsample(indice_node{ii},options.batch_size);
            % calculate gradient
            grad_node{ii}= problem.grad( w_node{ii},sam_indice_node{ii});
            
            w_node{ii} = w_consen_node{ii} - step * grad_node{ii};
            
            %proximal operator
            if prox==1
                w_node{ii}  = problem.prox(  w_node{ii} , step);
            end
            % increment total iteration numbers
        end
        parfor ii=1: num_nod
            w_consen_node{ii}=zeros(d,1);
            for jj=1:size(options.net,1)
                w_consen_node{ii} = w_consen_node{ii}+ options.net(ii,jj)*w_node{jj};
            end
        end
        w=mean(cell2mat(w_node),2);
                total_iter = total_iter + 1;

    end
    
    % measure elapsed time
    elapsed_time = toc(start_time);
    
    % count gradient evaluations
    grad_calc_count = grad_calc_count + j * ii*options.batch_size;
    epoch = epoch + 1;   
    
    % store infos
    [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);
    
    % display infos
    if options.verbose > 0
        fprintf('DSGD-%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
    end
end

if optgap < options.tol_optgap
    fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
elseif epoch == options.max_epoch
    fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
end
end

