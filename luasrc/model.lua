--
-- Created by IntelliJ IDEA.
-- User: asebag
-- Date: 1/13/17
-- Time: 7:35 PM
-- To change this template use File | Settings | File Templates.
--

require 'nn'

local function get_regression(dim_input)
    --Coding a simple regression
    local net = nn.Sequential()

    net:add(nn.View(dim_input))

    net:add(nn.Linear(dim_input, num_classes))

    return net
end

local function get_cifar_linear(dim_input, add_dropout, add_batchnorm)
    local lin_features1 = 1500
    local lin_features2 = 900

    local net = nn.Sequential()
    net:add(nn.View(dim_input))

    if add_dropout then net:add(nn.Dropout(0.8)) end -- check if using inplace dropout layers anything changes - I don't think it should
    net:add(nn.Linear(dim_input, lin_features1))
    if add_batchnorm then net:add(nn.BatchNormalization(lin_features1)) end
    net:add(nn.ReLU())                       -- non-linearity

    if add_dropout then net:add(nn.Dropout(0.5)) end
    net:add(nn.Linear(lin_features1, lin_features2))
    if add_batchnorm then net:add(nn.BatchNormalization(lin_features2)) end
    net:add(nn.ReLU())                       -- non-linearity

    if add_dropout then net:add(nn.Dropout(0.5)) end
    net:add(nn.Linear(lin_features2, num_classes))

    return net
end

local function get_mnist_linear(dim_input, add_dropout, add_batchnorm)
    local lin_features1 = 500
    local lin_features2 = 300

    local net = nn.Sequential()
    net:add(nn.View(dim_input))

    if add_dropout then net:add(nn.Dropout(0.8)) end
    net:add(nn.Linear(dim_input, lin_features1))
    if add_batchnorm then net:add(nn.BatchNormalization(lin_features1)) end
    net:add(nn.ReLU())                       -- non-linearity

    if add_dropout then net:add(nn.Dropout(0.5)) end
    net:add(nn.Linear(lin_features1, lin_features2))
    if add_batchnorm then net:add(nn.BatchNormalization(lin_features2)) end
    net:add(nn.ReLU())                       -- non-linearity

    if add_dropout then net:add(nn.Dropout(0.5)) end
    net:add(nn.Linear(lin_features2, num_classes))

    return net
end

local function get_cifar_AlexNet(add_dropout, add_batchnorm)

    local outputfeatures1 = 32
    local outputfeatures2 = 64
    local linearfeatures = 128*3

    local input_dropout_rate = 0.9
    local conv_dropout_rate = 0.75
    local lin_dropout_rate = 0.5

    local net = nn.Sequential()
    if add_dropout then net:add(nn.SpatialDropout(input_dropout_rate)) end
    net:add(nn.SpatialConvolution(3, outputfeatures1, 5, 5)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel
    if add_batchnorm then net:add(nn.SpatialBatchNormalization(outputfeatures1)) end
    net:add(nn.ReLU())                       -- non-linearity
    --net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
    net:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))


    if add_dropout then net:add(nn.SpatialDropout(conv_dropout_rate)) end
    net:add(nn.SpatialConvolution(outputfeatures1, outputfeatures2, 5, 5))
    if add_batchnorm then net:add(nn.SpatialBatchNormalization(outputfeatures2)) end
    net:add(nn.ReLU())                       -- non-linearity
    net:add(nn.SpatialMaxPooling(2,2,2,2))

    if add_dropout then net:add(nn.SpatialDropout(conv_dropout_rate)) end
    net:add(nn.View(outputfeatures2*3*3))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
    net:add(nn.Linear(outputfeatures2*3*3, linearfeatures))             -- fully connected layer (matrix multiplication between input and weights)
    if add_batchnorm then net:add(nn.BatchNormalization(linearfeatures)) end
    net:add(nn.ReLU())                       -- non-linearity

    if add_dropout then net:add(nn.Dropout(lin_dropout_rate)) end
    net:add(nn.Linear(linearfeatures, linearfeatures))
    if add_batchnorm then net:add(nn.BatchNormalization(linearfeatures)) end
    net:add(nn.ReLU())                       -- non-linearity

    if add_dropout then net:add(nn.Dropout(lin_dropout_rate)) end
    net:add(nn.Linear(linearfeatures, num_classes))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
    return net
end

local function get_mnist_AlexNet(add_dropout, add_batchnorm)
    --Changing SpatialConvolutionMM to SpatialConvolution, nn.Reshape to nn.View and adding batch normalizations
    local outputfeatures1 = 32
    local outputfeatures2 = 64
    local linear_features = 128

    local input_dropout_rate = 0.8
    local dropout_rate = 0.5

    local model = nn.Sequential()
  ------------------------------------------------------------
    -- convolutional network
    ------------------------------------------------------------
    -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
    if add_dropout then model:add(nn.SpatialDropout(input_dropout_rate)) end
    model:add(nn.SpatialConvolution(1, outputfeatures1, 5, 5))
    if add_batchnorm then model:add(nn.SpatialBatchNormalization(outputfeatures1)) end
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))

    -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
    if add_dropout then model:add(nn.SpatialDropout(dropout_rate)) end
    model:add(nn.SpatialConvolution(outputfeatures1, outputfeatures2, 5, 5))
    if add_batchnorm then model:add(nn.SpatialBatchNormalization(outputfeatures2)) end
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- stage 3 : standard 2-layer MLP:
    if add_dropout then model:add(nn.SpatialDropout(dropout_rate)) end
    model:add(nn.View(outputfeatures2*3*3))
    model:add(nn.Linear(outputfeatures2*3*3, linear_features))
    if add_batchnorm then model:add(nn.BatchNormalization(linear_features)) end
    model:add(nn.ReLU())

    if add_dropout then model:add(nn.Dropout(dropout_rate)) end
    model:add(nn.Linear(linear_features, linear_features))
    if add_batchnorm then model:add(nn.BatchNormalization(linear_features)) end
    model:add(nn.ReLU())

    if add_dropout then model:add(nn.Dropout(dropout_rate)) end
    model:add(nn.Linear(linear_features, num_classes))

    return model
end

local function get_mnist_net()
    local model = nn.Sequential()
  ------------------------------------------------------------
    -- convolutional network 
    ------------------------------------------------------------
    -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
    model:add(nn.SpatialConvolution(1, 32, 5, 5))
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))
    -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
    model:add(nn.SpatialConvolution(32, 64, 5, 5))
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    -- stage 3 : standard 2-layer MLP:
    model:add(nn.View(64*3*3))
    model:add(nn.Linear(64*3*3, 200))
    model:add(nn.Tanh())
    model:add(nn.Linear(200, num_classes))

    return model
end

function get_model(data, dim_input, model_type, add_batchnorm, add_dropout)
    local net
    local add_dropout = add_dropout or false

    if model_type==0 then
        net = get_regression(dim_input)

    elseif data=='cifar' then
        if model_type==4 then
            net = get_cifar_AlexNet()
        elseif model_type==2 then
            net = get_cifar_linear(dim_input, add_dropout, add_batchnorm)
        end

    elseif data=='mnist' then
        if model_type==4 then
            net= get_mnist_AlexNet()
        elseif model_type==2 then
            net = get_mnist_linear(dim_input, add_dropout, add_batchnorm)

        end
    end
    
    net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems
    local criterion = nn.ClassNLLCriterion()    
    criterion.sizeAverage = true -- to be used in batch mode

    local num_modules = #net.modules
--WEIGHT INIT
    for i=1,num_modules do
      local m = net.modules[i]
      local name = torch.type(m)

      if name:find('Convolution') then
          print("Init weights for ", name)

          m.weight:normal(0.0, 0.02)
          m.bias:fill(0) -- Conv layers do have biases

      elseif name:find('Linear') then
         print("Init weights for ", name)

         local u = math.sqrt(6/(m.weight:size(1)+m.weight:size(2)))
         m.weight:uniform(-u, u)
         if i~=num_modules-1 then m.bias:fill(0)
                             else m.bias:fill(1/10) end

      elseif name:find('BatchNormalization') then
         print("Init weights for ", name)

         if m.weight then m.weight:normal(1.0, 0.02) end
         if m.bias then m.bias:fill(0) end
      end
    end
    
    --EVALUATE OR TRAINING MODE
    function net:evaluate()
        --Need to define this as it is not defined in nn
        local numModules = #self.modules
        for i=1,numModules do
            pcall(function () self.modules[i]:evaluate() return 1 end)
        end
    end

    function net:training()
        --Need to define this as it is not defined in nn
        local numModules = #self.modules
        for i=1,numModules do
            pcall(function() self.modules[i]:training() return 1 end)
        end
    end
    net:training()--It should already be the case but never too cautious
    
    return {net, criterion}
end
