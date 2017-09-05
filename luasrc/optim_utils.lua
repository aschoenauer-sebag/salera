--
-- Created by IntelliJ IDEA.
-- User: asebag
-- Date: 1/13/17
-- Time: 8:31 PM
-- To change this template use File | Settings | File Templates.
--

require 'image'
require 'gnuplot'
require 'pl'

function setSeed(seed)
   torch.setnumthreads(threads)
   math.randomseed(seed)
   torch.manualSeed(seed)
   if cutorch then
       cutorch.manualSeed(seed)
   end
   print("Setting random seed for CPUs and GPUs, setting num threads on CPUs")
end


local function write_single_output(folder,filename, value)
    local file = io.open(path.join(folder, filename), 'a')
    file:write(value ..'\n')
    file:close()
end

function write_verbose_output(outputfolder, i, norm_grad,
                        relaxed_sum_norm,
                        curr_lr, lr_std, adapt_factor)

    write_single_output(outputfolder, 'Gradient_'..i..'.txt', norm_grad)

    local file = io.open(path.join(outputfolder, 'LearningRate_'..i..'.txt'), 'a')
    if lr_std then
        file:write(curr_lr .. " ".. lr_std .. '\n')
    else
        file:write(curr_lr ..'\n')
    end
    file:close()

    if relaxed_sum_norm then
        local file = io.open(path.join(outputfolder, 'RelaxedSum_'..i..'.txt'), 'a')
        file:write(relaxed_sum_norm ..'\n')
        file:close()
    end

    if adapt_factor then
        local file = io.open(path.join(outputfolder, 'Adaptfactor_'..i..'.txt'), 'a')
        file:write(adapt_factor ..'\n')
        file:close()
    end

end


function write_error_output(outputfolder, train_error, test_error)
    write_single_output(outputfolder, 'Trainerr.txt', train_error)
    write_single_output(outputfolder, 'Testerr.txt', test_error)
end

function save_model_full_test_error(outputfolder, t, testError, module)
    --i. Save full test error
    write_single_output(outputfolder, 'FullTesterr.txt', t..' '.. testError)

    --ii. Save model
    if save_model then
        local mod
        if module~=nil then
            mod = module:clone()
            mod:evaluate()-- Thus I save the model in evaluation mode,
                    --ie for example dropout layers will be identity
            mod:double() --Also default save it on CPU and not GPU because cannot unload it from there otherwise
            mod:clearState()
        end

        torch.save(outputfolder .. '/model_' .. t .. '.bin', mod)
        mod = nil; collectgarbage()
    end
end

function get_gradient_dimensions(module)
    local gradient_dimensions = {}
    for i=1,#module.modules do
        if module.modules[i].weight then
            local dim = module.modules[i].weight:nElement()
            if module.modules[i].bias then dim = dim + module.modules[i].bias:nElement() end
            table.insert(gradient_dimensions, dim)
        end
    end
    return torch.Tensor(gradient_dimensions)
end

local utils = {
    write_verbose_output = write_verbose_output,
    write_error_output = write_error_output,
    get_gradient_dimensions = get_gradient_dimensions,
    save_gradient_weights = save_gradient_weights,
    save_model_full_test_error = save_model_full_test_error,
    setSeed = setSeed
}
return utils


