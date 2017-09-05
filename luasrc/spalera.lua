--[[ An implementation of SPALeRA SGD (Schoenauer-Sebag et al., 2017), as desribed in the paper

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.weightDecay'       : weight decay

S-AgMA PARAMETERS
    - 'config.adapt_constant'     : parameter C
    - 'config.alpha'              : parameter alpha (memory parameter of AgMA)
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

function spalera(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local wd = config.weightDecay or 0
   local adapt_constant = config.adapt_constant
   local alpha = config.alpha
   local plotinterval = config.plotinterval
   local gLambda = config.SALeRA_lambda or 0.1

   local mbratio = config.mbratio

   local gradient_dimension = config.gradient_dimensions:sum()

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
       print("Starting lrs and relaxed sums")
       local typename = torch.type(x)
       -- Parameter-wise learning rates start at 1
       state.learningRates = x.new(dfdx:size()):fill(1)
       if string.find(typename, 'Cuda') then
           state.learningRates = state.learningRates:cudaDouble()
       else
           state.learningRates = state.learningRates:double()
       end

       state.relaxed_sums = x.new(dfdx:size()):zero()

       print("Starting Page Hinkley")
       --Init of lambda parameter
       print('Using gLambda ', gLambda)
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
       local paramref_mean = REF_transitory(alpha, eveCounter +1)/gradient_dimension
       local paramref_std = std_deviation(alpha, gradient_dimension)/math.sqrt(gradient_dimension)

       local lrs = state.learningRates
       local relaxed_sums = state.relaxed_sums

       update_relaxed_sum(dfdx, relaxed_sums, alpha)
       local sq_relaxed_sum = relaxed_sums:clone():pow(2)
       local adapt = sq_relaxed_sum:add(-paramref_mean):mul(adapt_constant/paramref_std):exp()
       lrs:cmul(adapt)

        -- (5) parameter update with single or individual learning rates
        if not state.deltaParameters then
            state.deltaParameters = x.new(dfdx:size())
        end
        state.deltaParameters:copy(lrs):cmul(dfdx)
        state.previous_x = x:clone()
        x:add(-lr, state.deltaParameters)

        --Now checking if the error shows a tremendous increase - abnormal according to the Page-Hinkley test
       -- update evaluation counter
        state.EveCounter = state.EveCounter + 1
        state.PHCounter = state.PHCounter +1

       -- return x*, f(x) before optimization
       return x,fx
   end

end

