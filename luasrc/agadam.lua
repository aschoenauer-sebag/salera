--[[ An implementation of Adam (Kingma et Ba, 2015) with AgMA (Schoenauer-Sebag et al., 2017) for adapting the learning rates

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- `config.learningRateDecay` : learning rate decay
- 'config.weightDecay'       : weight decay
ADAM PARAMETERS
    - 'config.beta1'             : first moment coefficient
    - 'config.beta2'             : second moment coefficient
    - 'config.epsilon'           : for numerical stability
AgMA PARAMETERS
    - 'config.adapt_constant'     : parameter C
    - 'config.alpha'              : parameter alpha (memory parameter of AgMA)
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

function agadam(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 0.001

   local beta1 = config.beta1 or 0.9
   local beta2 = config.beta2 or 0.999

   local adapt_constant = config.adapt_constant
   local alpha = config.alpha or 0.001

   local epsilon = config.epsilon or 1e-8
   local wd = config.weightDecay or 0
   
   local individual_dimensions = config.gradient_dimensions
   local outputfolder = config.outputfolder
   local plotinterval = config.plotinterval

   -- (1) evaluate f(x) and df/dx
   local fx, dfdx = opfunc(x)

   -- (2) weight decay
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   -- Initialization
   state.t = state.t or 0
   -- Exponential moving average of gradient values
   state.m = state.m or x.new(dfdx:size()):zero()
   -- Exponential moving average of squared gradient values
   state.v = state.v or x.new(dfdx:size()):zero()
   -- A tmp tensor to hold the sqrt(v) + epsilon
   state.denom = state.denom or x.new(dfdx:size()):zero()

   -- need to do initialization if not been done before
   if state.t==0 then
       print("Starting lrs and relaxed sums")
       state.learningRates = torch.Tensor(dfdx:size()):fill(lr)
       state.relaxed_sums = torch.Tensor(dfdx:size(1)):zero()
   end

   local lrs = state.learningRates
   local relaxed_sums = state.relaxed_sums

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
       update_relaxed_sum(curr_grad, curr_relaxed_gradient_sum, alpha) -- looking at alpha = 0.01

       local relaxed_sum_norm = norm_l2(curr_relaxed_gradient_sum)^2
       local ref_mean = REF_transitory(alpha, state.t+1)
       local ref_std = std_deviation(alpha, dimension)

       local adapt = math.exp(adapt_constant/ref_std*(relaxed_sum_norm - ref_mean))
       curr_lr = curr_lr*adapt

       --Updating values
       lrs[{{curr_dim+1, curr_dim+dimension}}]=curr_lr

       if config.verbose and math.fmod(state.t, plotinterval)==0 then
           utils.write_verbose_output(outputfolder, i, curr_grad_norm, relaxed_sum_norm,
                                       curr_lr,
                                       0,
                                       adapt
           )
       end
       curr_dim= curr_dim+dimension
   end

   state.t = state.t + 1

   -- Decay the first and second moment running average coefficient
   state.m:mul(beta1):add(1-beta1, dfdx)
   state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

   state.denom:copy(state.v):sqrt():add(epsilon)

   local biasCorrection1 = 1 - beta1^state.t
   local biasCorrection2 = 1 - beta2^state.t
   local step = lrs:clone():mul(math.sqrt(biasCorrection2)/biasCorrection1):cmul(state.m):cdiv(state.denom)
   -- (4) update x
   x:add(-step)

   -- return x*, f(x) before optimization
   return x, {fx}

end
