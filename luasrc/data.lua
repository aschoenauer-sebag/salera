--
-- Created by IntelliJ IDEA.
-- User: asebag
-- Date: 1/13/17
-- Time: 7:35 PM
-- To change this template use File | Settings | File Templates.
--

local function _getBatch(dataset, batchsize, iter, shuffledIndices)
   local data = dataset.data
   local targets = torch.Tensor(batchsize):typeAs(data)

   local sample_ = data[1]
   local inputs 
   if sample_:size():size()>1 then
      inputs = torch.Tensor(batchsize, sample_:size(1), sample_:size(2), sample_:size(3)):typeAs(data)
   else
      inputs = torch.Tensor(batchsize, sample_:size(1)):typeAs(data)
   end

   for i = 1,batchsize do
        -- load new sample
       local indice = math.fmod(iter*batchsize + i,data:size(1))
       if indice==0 then indice = data:size(1) end
       local sample = dataset[shuffledIndices[indice]]

       inputs[i]=sample[1]:clone()
       targets[i]= sample[2]
   end

   return inputs, targets
end

mnist = {}

mnist.path_dataset ='mnist.t7'
mnist.path_trainset = paths.concat(mnist.path_dataset, 'train_32x32.t7')
mnist.path_testset = paths.concat(mnist.path_dataset, 'test_32x32.t7')

function mnist.loadTrainTest()
    --Looks if you have MNIST data under Torch format in your current directory, and if not, downloads it.
    if not paths.dirp(mnist.path_dataset) or not paths.filep(mnist.path_trainset) then
        os.execute('wget -c https://s3.amazonaws.com/torch7/data/mnist.t7.tgz')
        os.execute('tar xvf mnist.t7.tgz')
    end

    -- create training set and normalize
    local trainset = mnist.loadDataset(mnist.path_trainset)
    local mean, std = trainset:normalizeGlobal()
    
    -- create test set and normalize
    local testset = mnist.loadDataset(mnist.path_testset)
    testset:normalizeGlobal(mean, std)
    
    return trainset, testset
end

function mnist.loadDataset(fileName)
   local f = torch.load(fileName, 'ascii')
   local data = f.data:type(torch.getdefaulttensortype())
   local nExample = f.data:size(1)

   local labels = torch.Tensor(nExample)
   labels:copy(f.labels)

   local shuffledIndices = torch.randperm(nExample, 'torch.LongTensor')
   print('<mnist> done')

   local dataset = {}
   dataset.data = data
   dataset.label = labels

   function dataset:normalizeGlobal(mean_, std_)
      print('Normalizing MNIST')
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   function dataset:size()
      return nExample
   end

   setmetatable(dataset, {__index = function(self, index)
           local input = self.data[index]
           local class = self.label[index]
           local example = {input, class}
                       return example
   end})
   
   function dataset:getBatch(batchsize, iter)
       if math.fmod(batchsize*iter, self:size())==0 then
           --Re-shuffling indices if it is a new epoch
           shuffledIndices = torch.randperm(nExample, 'torch.LongTensor')
       end
       return _getBatch(dataset, batchsize, iter, shuffledIndices)
   end

   return dataset
end

cifar ={}
cifar.path_trainset = 'cifar10-train.t7'
cifar.path_testset = 'cifar10-test.t7'


function cifar.loadTrainTest()
    --Looks if you have the cifar10 data under Torch format in your current directory, and if not, downloads it.
    if not (paths.filep("cifar10torch.zip") or paths.filep(cifar.path_trainset)) then
        os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torch.zip')
        os.execute('unzip cifar10torch.zip')
    end

    --Labels need to be between 1 and num_classes.
    local trainset = torch.load(cifar.path_trainset)
    local testset = torch.load(cifar.path_testset)
   
    --print(trainset.data:size())-- 50 000, 3, 32, 32

    function trainset:size()
        return self.data:size(1)
    end
    function testset:size()
        return self.data:size(1)
    end

    local shuffledTrainIndices = torch.randperm(trainset:size(), 'torch.LongTensor')
    local shuffledTestIndices = torch.randperm(testset:size(), 'torch.LongTensor')

    setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]+1}
                end}
    );

    setmetatable(testset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]+1}
                end}
    );

    function trainset:getBatch(batchsize, iter)
        if math.fmod(batchsize*iter, self:size())==0 then
           shuffledTrainIndices = torch.randperm(self:size(), 'torch.LongTensor')
       end
       return _getBatch(self, batchsize, iter, shuffledTrainIndices)
    end

    function testset:getBatch(batchsize, iter)
       if math.fmod(batchsize*iter, self:size())==0 then
           --Re-shuffling indices if it is a new epoch
           shuffledTestIndices = torch.randperm(self:size(), 'torch.LongTensor')
       end
       return _getBatch(self, batchsize, iter, shuffledTestIndices)
    end

    trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
    testset.data = testset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

    if gpu then
        trainset.data = trainset.data:cudaDouble()
        testset.data = testset.data:cudaDouble()
    end

    local mean = {} -- store the mean, to normalize the test set in the future
    local stdv  = {} -- store the standard-deviation for the future
    for i=1,3 do -- over each image channel
        mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
        trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

        stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
        trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
        testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end
    print("Data normalization done")
    return trainset, testset

end