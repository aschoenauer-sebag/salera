--[[ An implementation of ALeRA SGD (Schoenauer-Sebag et al., 2017)

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.weightDecay'       : weight decay

ALeRA PARAMETERS
    - 'config.adapt_constant'     : parameter C
    - 'config.alpha'              : parameter alpha (memory parameter of ALeRA)
    - 'config.gradient_dimensions': a torch.Tensor which contains the dimension of the gradient for each layer (torch.Tensor of shape num_layers)

- `config.verbose` : indicates if you want to save gradient norm, mean(learning_rate) and std(learning_rate) for each layer. If yes, provide:
    - `config.outputfolder`: where to save it
    - `config.plotinterval`: when to save these informations

- 'state'                    : a table describing the state of the optimizer; after each
                              call the state is modified

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

function alera(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local outputfolder = config.outputfolder
   local plotinterval = config.plotinterval

   local lr = config.learningRate or 1e-3
   local wd = config.weightDecay or 0

   local adapt_constant = config.adapt_constant
   local alpha = config.alpha or 0.001
   local individual_dimensions = config.gradient_dimensions

   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)
   -- (2) weight decay with single parameter
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   -- need to do initialization if not been done before
   if nevals==0 then
       print("Starting lrs and relaxed sums")
       local typename = torch.type(x)
       state.learningRates = x.new(dfdx:size()):fill(lr)
       if string.find(typename, 'Cuda') then
          state.learningRates = state.learningRates:cudaDouble()
       else
          state.learningRates = state.learningRates:double()
       end

       state.relaxed_sums = x.new(dfdx:size(1)):zero()
   end

   local lrs = state.learningRates
   local relaxed_sums = state.relaxed_sums

   -- (2) weight decay with single or individual parameters
        -- TODO This seems the right thing to me
   if wd ~= 0 then
      dfdx:add(wd, x)
   elseif wds then
      if not state.decayParameters then
         state.decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.decayParameters:copy(wds):cmul(x)
      dfdx:add(state.decayParameters)
   end

   local curr_dim = 0
   local numModulesWithLearnableParameters = individual_dimensions:size(1)
    --Looking at the gradients for each layer one by one
   for i=1,numModulesWithLearnableParameters do
       --This is the dimension of the gradient for module i
       local dimension = individual_dimensions[i]
        --This is the current learning rate for this module
       local curr_lr = lrs[curr_dim+1]

        --This is the new gradient for these parameters
       local curr_grad = dfdx[{{curr_dim+1, curr_dim+dimension}}]
       local curr_grad_norm = norm_l2(curr_grad)

       local curr_relaxed_gradient_sum = relaxed_sums[{{curr_dim+1, curr_dim+dimension}}]

        --I update the relaxed sum of past gradients with this new gradient
       update_relaxed_sum(curr_grad, curr_relaxed_gradient_sum, alpha)

       local relaxed_sum_norm = norm_l2(curr_relaxed_gradient_sum)^2
        -- What the random value of the relaxed sum would be
       local ref_mean = REF_transitory(alpha, nevals+1)
       local ref_std = std_deviation(alpha, dimension)

       local adapt = math.exp(adapt_constant/ref_std*(relaxed_sum_norm - ref_mean))
       curr_lr = curr_lr*adapt

       --Updating values
       lrs[{{curr_dim+1, curr_dim+dimension}}]=curr_lr

       if config.verbose and math.fmod(nevals, plotinterval)==0 then
           utils.write_output_lear(outputfolder, i, curr_grad_norm, relaxed_sum_norm,
                                       0,
                                       0,
                                       curr_lr,
                                       0,
                                       adapt
           )
       end
       curr_dim= curr_dim+dimension
   end

    -- (5) parameter update with single or individual learning rates
    if not state.deltaParameters then
        state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
    end
    state.deltaParameters:copy(lrs):cmul(dfdx)
    x:add(-1, state.deltaParameters)

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,fx
end

