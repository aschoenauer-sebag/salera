--
-- Created by IntelliJ IDEA.
-- User: asebag
-- Date: 1/13/17
-- Time: 7:21 PM
-- To change this template use File | Settings | File Templates.
--

-- Dataset characteristics
classes = {'airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck' }
num_classes = #classes


-- Net parameters
kernelsize1 = 7
kernelsize2 = 5
outputfeatures = 64
linear_features = 200
dropout_rate = 0.5