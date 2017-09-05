--
-- Created by IntelliJ IDEA.
-- User: asebag
-- Date: 1/13/17
-- Time: 7:20 PM
-- To change this template use File | Settings | File Templates.
--
if gpu then require 'cunn' end

require 'model'
require 'data'
utils = require 'optim_utils'

require 'salera_utils'
--Optimization options
require 'optim'
require 'agadam'
require 'alera'
require 'salera'
require 'spalera'

---Global parameters
--Number of epochs for which to train
num_epochs = 20
--Lower bound on gradient for updating the relaxed sum
epsilon = 1e-6
--Number of test samples for a rapid evaluation of test error during learning
testingBatch = 1000

 --------------------------------------------------------------------
 -- computes statistics / error
 --

local function getError(dataset, module, iter)
    --Using the built-in ConfusionMatrix from the optim package, which should hopefully be faster

    module:evaluate()

    local testing_err=0

    local testingInputs, testingTargets
    if iter~='full' then
        testingInputs, testingTargets = dataset:getBatch(testingBatch, iter)
    else
        testingInputs, testingTargets = dataset.data, dataset.label
    end

    local predictions = module:forward(testingInputs)
    local num_evaluations = predictions:size(1)
    for i=1,num_evaluations do
        local currPred = predictions[i]
        _, currPred = torch.max(currPred, 1)
        currPred = currPred[1]

        if currPred~=testingTargets[i] then
            testing_err=testing_err+1
        end
    end
    testing_err = testing_err/num_evaluations

    module:training()
    return testing_err
end


 --------------------------------------------------------------------
 -- MAIN FUNCTION
 --
--[[
ARGS:

- 'outputfolder' : full name of folder where all results will be saved

    OPTIMIZATION PARAMETERS
    - 'data'                     : 'mnist' or 'cifar' for MNIST or CIFAR-10
    - 'optim_func_name'          : possibilities:
                            * 'agadam'      for Ag-Adam
                            * 'alera'        for ALeRA
                            * 'salera'       for S-AgMA
                            * 'spalera'      for parameter-wise S-AgMA
                            * 'adam'        for Adam
                            * 'nesterov'         for NAG
                            * 'adagrad'     for Adagrad
    - 'hyperparam1'              : float, parameter alpha for (S-)ALeRA/AgAdam/S-PAgMA, momentum for NAG and beta1 for Adam
    - 'hyperparam2'              : float, parameter C for (S-)ALeRA/S-PAgMA and beta2 for Adam
    - 'eta'                      : float, initial learning rate

    - 'model_type'               : int, either 0, 2 or 4 for models as in the paper
    - 'mb_ratio'                 : the ratio of mini-batch size to training set size (for Page-Hinkley relaxed loss computations)
    - 'add_batchnorm'            : either true or leave blank for adding batch normalization layers or not

    - 'gLambda'                  : modify the divisor of the first mini-batch loss, whose ratio defines the alarm threshold for S-AgMA and S-PAgMA
    - 'seed'                     : int/float, random seed for this run

RETURN:
nothing, saves the test error in a text file in the outputfolder.
]]

function train(outputfolder, data, optim_func_name, hyperparam1, hyperparam2, eta,
                        model_type, mb_ratio,
                        add_batchnorm, gLambda, seed
)
    local seed = seed or 5
    local gLambda = gLambda or 0.1

    if data=='cifar' then
        require 'settings_cifar'
    elseif data=='mnist' then
        require 'settings_mnist'
    end

    outputfolder = outputfolder..optim_func_name ..model_type ..'m'..mb_ratio
    if model_type==2 or model_type==4 then outputfolder = outputfolder .. 'bn'..tostring(add_batchnorm==true) end
    outputfolder = outputfolder ..data..'seed'..seed..'p'.. hyperparam1 .. 'pp'.. hyperparam2 ..'l'..gLambda..'e'..eta

    print("Working in ", outputfolder)
    if not paths.dirp(outputfolder) then
        paths.mkdir(outputfolder)
    end

    --SET SEEDS
    utils.setSeed(seed)
    --SET TORCH TENSOR as CudaDouble if using the GPU
    --Careful, apparently this is bad practice according to ppl who developed Torch
        --In particular, it will break the code if trying to load raw images (but works fine in the case at hand)
    if gpu then
        torch.setdefaulttensortype('torch.CudaDoubleTensor')
    end

    local model_type = model_type or 4

    --LOAD DATA
    local trainset, testset
    local groundtruth_testset
    if data=='cifar' then
        trainset, testset = cifar.loadTrainTest()
    elseif data=='mnist' then
        trainset, testset = mnist.loadTrainTest()
    end

    local datadimension = trainset[1][1]:nElement()
    local datasize = trainset:size()
    local batchsize = datasize *mb_ratio
    local maxiter = num_epochs / mb_ratio + 5

    local shortInterval = 1/mb_ratio
    print('Going for ', maxiter, ' iterations with ', batchsize, ' as batch size.')

    --LOAD MODEL
    local module = get_model(data, datadimension, model_type, add_batchnorm)

    local criterion = module[2]
    local module = module[1]
    if gpu_cudnn then
        require 'cudnn'
        cudnn.fastest=true; cudnn.benchmark =true
        cudnn.convert(module, cudnn)
    end

    --CHOOSE OPTIMIZATION ALGORITHM AS SPECIFIED
    local beta1 = hyperparam1
    local beta2 = hyperparam2
    local optim_func
    if optim_func_name=='adagrad' then
        optim_func=optim.adagrad
    elseif optim_func_name=='adam' then
	    optim_func=optim.adam
    elseif optim_func_name=='nesterov' then
        optim_func = optim.nag
    elseif optim_func_name=='agadam' then
        optim_func = agadam
        beta1 = 0.9
        beta2 = 0.999
    elseif optim_func_name=='alera' then
        optim_func = alera
    elseif optim_func_name=='salera' then
        optim_func = salera
    elseif optim_func_name=='spalera' then
        optim_func = spalera
    end
     
    -- get all parameters
    x,dl_dx= module:getParameters()

    --get the dimension of each layer
    local gradient_dimensions = utils.get_gradient_dimensions(module)
    assert(dl_dx:size(1)==gradient_dimensions:sum())
    ----------------------------------------------------------------------
    -- train model
    --
    local iter = 0
    for t = 1,maxiter do
    --------------------------------------------------------------------
       -- create mini-batch
       --
       local inputs, targets = trainset:getBatch(batchsize, iter)
       --------------------------------------------------------------------
       -- progress
       --
       iter = iter+1
       xlua.progress(iter, maxiter)

       --------------------------------------------------------------------
       -- define eval closure
       --
       local feval_batch = function()
          -- reset gradient/f
          module:zeroGradParameters()
          local predictions = module:forward(inputs)
          local currErr =criterion:forward(predictions, targets)

          -- gradients
          local gradInput = criterion:backward(module.output, targets)
          module:backward(inputs, gradInput)

          -- return f and df/dx
          return currErr,dl_dx
       end

       --------------------------------------------------------------------
       -- one SGD step
       --
       if iter==1 then
           sgdconf = {
                       --Generic parameters
                       learningRate = eta,

                       --Nesterov parameters
                       momentum = hyperparam1,
                       
                       --Adam parameters
                       beta1 = beta1,
                       beta2 = beta2,
      
                       --SALeRA parameters
                       SALeRA_lambda = gLambda,
                       alpha = hyperparam1,
                       adapt_constant = hyperparam2, -- for Eve, ie with adaptive multiplicative factor for learning rate
                       gradient_dimensions = gradient_dimensions,

                       --Parameters for saving information during training
                       outputfolder = outputfolder,
                       verbose = verbose,
                       plotinterval=shortInterval,
                       mbratio = mb_ratio,
                       gpu=gpu}
       end  
                             
       --Doing stochastic gradient step
       _,fs = optim_func(feval_batch, x, sgdconf)

       --Checking if the gradient norm is NaN or learning rate very small
       local gradient_norm = norm_l2(dl_dx)
       if math.fmod(t-1, 10000)==0 then print(t, fs, gradient_norm) end
       local _gradient_norm = torch.Tensor({gradient_norm})
       local timepoints = torch.Tensor({5, 10, 20, 30, 40, 50, 60, 70, 80})

       --If gradient norm has become NaN, saving an error of 100% before stopping
       if _gradient_norm:ne(_gradient_norm)[1]==1 then
           print('Gradient becomes NaN, stopping run here')
           for ind=1,9 do
               if t<=timepoints[ind]/mb_ratio+1 then
                   utils.save_model_full_test_error(outputfolder, timepoints[ind], 1, nil)
               end
           end
           break

       --If learning rate has become too small as reported by the optim algorithm, see the test error and save it before stopping
       elseif fs==false then
           print('Learning rate <1e-15, stopping run here')
           print('Looking at full test error before stopping.')
           local fullTestErr = getError(testset, module, 'full')
           for ind=1,9 do
               if t<=timepoints[ind]/mb_ratio+1 then
                   utils.save_model_full_test_error(outputfolder, timepoints[ind], fullTestErr, nil)
               end
           end
           break
       end

       --LOOKING AT TRAIN AND TEST ERRORS, on a SUBSET of train and test sets
       if math.fmod(t-1, shortInterval)==0 then
          local trainErr = getError(trainset, module, iter)
          local testErr = getError(testset, module, iter)

          utils.write_error_output(outputfolder, trainErr, testErr)
       end

        --LOKING AT FULL TEST ERROR FOR FINAL RESULT (model is not saved)
        if t==1 or t-1==5/mb_ratio
                or t-1==10/mb_ratio
                or t-1==20/mb_ratio
                or t-1==30/mb_ratio
                or t-1==40/mb_ratio
                or t-1==50/mb_ratio
                or t-1==60/mb_ratio
                or t-1==70/mb_ratio
                or t-1==80/mb_ratio
        then
            print('Looking at full test error.')
            local fullTestErr = getError(testset, module, 'full')
            utils.save_model_full_test_error(outputfolder, (t-1)*mb_ratio, fullTestErr, module)
        end

    end
end







