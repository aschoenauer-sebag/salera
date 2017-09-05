--[[ An implementation of SALeRA SGD (Schoenauer-Sebag et al., 2017), as desribed in the paper

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.weightDecay'       : weight decay

SALeRA PARAMETERS
    - 'config.adapt_constant'     : parameter C
    - 'config.alpha'              : parameter alpha (memory parameter of ALeRA)
    - 'config.gradient_dimensions': a torch.Tensor which contains the dimension of the gradient for each layer (torch.Tensor of shape num_layers)
    - 'config.mbratio'            : the ratio of mini-batch size to training set size (for Page-Hinkley relaxed loss computations

- `config.verbose` : indicates if you want to save gradient norm, mean(learning_rate) and std(learning_rate) for each layer. If yes, provide:
    - `config.outputfolder`: where to save it
    - `config.plotinterval`: when to save these informations

- 'state'                    : a table describing the state of the optimizer; after each
                              call the state is modified

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

require 'salera_utils'

function salera(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local outputfolder = config.outputfolder
   local lr = config.learningRate or 1e-3
   local wd = config.weightDecay or 0
   local adapt_constant = config.adapt_constant
   local alpha = config.alpha
   local gLambda = config.SALeRA_lambda or 0.1

   local mbratio = config.mbratio

   local individual_dimensions = config.gradient_dimensions
   local numModulesWithLearnableParameters = individual_dimensions:size(1)

   --Eval counter for PH
   state.PHCounter = state.PHCounter or 0
   --Eval counter for Eve
   state.EveCounter = state.EveCounter or 0

   local eveCounter = state.EveCounter
   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (2) weight decay with single parameter
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   -- initialization
   if not state.learningRates then
       print('Starting with weight decay ', wd, ' alpha ', alpha, ' C ', adapt_constant, ' PH gLambda ', gLambda)
       local typename = torch.type(x)
       state.learningRates = x.new(numModulesWithLearnableParameters):fill(lr)
       if string.find(typename, 'Cuda') then
           state.learningRates = state.learningRates:cudaDouble()
       else
           state.learningRates = state.learningRates:double()
       end

       state.relaxed_sums = x.new(dfdx:size()):zero()

       --Init of lambda parameter
       state.lambda = fx*gLambda
       init_PH(state)
   end
    --Looking to see if alert, can never be triggered at first iteration
   local alert = perform_PH_iter(fx, state)
   if alert then
       print('Alert ', eveCounter, state.PHCounter)
       --Backtracking
       x = state.previous_x
       --Dividing learning rates by 2
       state.learningRates:div(2)

       if torch.le(state.learningRates,1e-15):sum()>0 then
          print('Stopping because learning rate too low')
          return x, false
       end

       --RE-INIT EVE AND PH SUMS
       re_init_after_layer_explosion(state)
   else
       --Saving the last update
       state.previous_x = x:clone()

       local lrs = state.learningRates
       local relaxed_sums = state.relaxed_sums

       local curr_dim = 0
        --Looking at the gradients for each layer one by one
       for i=1,numModulesWithLearnableParameters do
           --This is the dimension of the gradient for layer i
           local dimension = individual_dimensions[i]
            --This is the current learning rate for layer i
            --This is the gradient for this layer
           local curr_grad = dfdx[{{curr_dim+1, curr_dim+dimension}}]
           local curr_grad_norm = norm_l2(curr_grad)

           local curr_relaxed_gradient_sum = relaxed_sums[{{curr_dim+1, curr_dim+dimension}}]
            --I update the relaxed sum of past gradients with this new gradient
           update_relaxed_sum(curr_grad, curr_relaxed_gradient_sum, alpha)
           local relaxed_sum_norm = norm_l2(curr_relaxed_gradient_sum)^2

            -- What the random value of the relaxed sum would be
           local ref_mean = REF_transitory(alpha, eveCounter +1)
           local ref_std = std_deviation(alpha, dimension)
           local adapt = math.exp(adapt_constant/ref_std*(relaxed_sum_norm - ref_mean))
           --Updating values
           lrs[i]=lrs[i]*adapt
           x[{{curr_dim+1, curr_dim+dimension}}]:add(-lrs[i], curr_grad)

           if config.verbose and math.fmod(eveCounter, config.plotinterval)==0 then
               utils.write_verbose_output(outputfolder, i, curr_grad_norm, relaxed_sum_norm,
                                           lrs[i],
                                           0,
                                           adapt )
           end

           curr_dim= curr_dim+dimension
       end

        -- update evaluation counter
        state.EveCounter = state.EveCounter + 1
        state.PHCounter = state.PHCounter +1

       -- return x*, f(x) before optimization
       return x,{fx}
   end

end

