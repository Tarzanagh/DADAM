function step = stepsize_alg(iter, options)
% extract options
if ~isfield(options, 'step_init')
    step_init = 0.1;
else
    step_init = options.step_init;
end

if ~isfield(options, 'step_alg')
    step_alg = 'fix';
else
    if strcmp(options.step_alg, 'decay')
        step_alg = 'decay';
    elseif strcmp(options.step_alg, 'decay-2')
        step_alg = 'decay-2';
    elseif strcmp(options.step_alg, 'decay-3')
        step_alg = 'decay-3';
    elseif strcmp(options.step_alg, 'decay-sq')
        step_alg = 'decay-sq';
    elseif strcmp(options.step_alg, 'fix')
        step_alg = 'fix';
    else
        step_alg = 'decay';
    end
end

if ~isfield(options, 'lambda')
    lambda = 0.1;
else
    lambda = options.lambda;
end


% update step-size
if strcmp(step_alg, 'fix')
    step = step_init;
elseif strcmp(step_alg, 'decay')
    step = step_init / (1 + step_init * lambda * iter);
elseif strcmp(step_alg, 'decay-2')
    step = step_init / (1 + iter);
elseif strcmp(step_alg, 'decay-3')
    step = step_init / (lambda + iter);
elseif strcmp(step_alg, 'decay-sq')
    step = step_init / sqrt(1 + step_init * lambda * iter);
end

end

