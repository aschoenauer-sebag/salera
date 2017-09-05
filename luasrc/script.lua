--
-- Created by IntelliJ IDEA.
-- User: lalil0u
-- Date: 10/02/17
-- Time: 15:50
-- To change this template use File | Settings | File Templates.
--
require 'main'
assert(os.setlocale("C"))
--Parameters to change depending on your setup
--Using the GPU?
gpu = false
--Number of threads for running on CPUs
threads = 10
--Having CUDNN drivers and package installed?
gpu_cudnn=false
--Willing to save the models as learning?
save_model = false
--Willing to record learning rates per layer, as well as gradient norm per layer during training?
verbose = false
--Folder where to write run results
local folder = 'Result_'

local data = arg[1]
local algorithm = arg[2]
local hyperparam1 = tonumber(arg[3])
local hyperparam2 = tonumber(arg[4])
local eta0 = tonumber(arg[5])

local model_type = 0
local mb_ratio = 0.01

if arg[6] then model_type = tonumber(arg[6]) end
if arg[7] then mb_ratio = tonumber(arg[7]) end

local add_batchnorm
if arg[8] and arg[8]~='-1' then add_batchnorm = true end

print("Let's go!")
train(folder, data, algorithm, hyperparam1, hyperparam2, eta0,
                    model_type, mb_ratio, add_batchnorm)

