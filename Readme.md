# SGD: Going as Fast as Possible, But Not Faster

This repository contains the code for the deep learning optimization algorithms described in "SGD: Going as Fast as Possible, But Not Faster" (Alice Schoenauer-Sebag, Marc Schoenauer and Michèle Sebag, [https://arxiv.org/abs/1709.01427]).

## Dependencies
* Code: the only dependency is Torch, which can be installed following the instructions here: http://torch.ch/docs/getting-started.html

* Data: if MNIST or CIFAR-10 datasets are not in the running repository, they will be downloaded and unzipped prior to running.

## Run
It is easy to run using the command line:
```
$ cd [folder_where_repo_was_cloned]
$ qlua -lenv script.lua [dataset] [algorithm] [hyper-parameter 1] [hyper-parameter 2] [learning rate] [model] [batch normalization]
```

Parameters:
- dataset: str. 
Possible values: 'mnist' or 'cifar'
- algorithm: str. 
Possible values: 'agadam', 'alera', 'salera', 'spalera' (paper agma-based methods), or 'nesterov' for NAG, 'adam' for Adam and 'adagrad' for Adagrad
- hyper-parameter 1: float. Value for the alpha parameter of all agma-based methods, momentum for NAG and beta1 for Adam.
- hyper-parameter 2: float. Value for the C factor of all agma-based methods, beta2 for Adam.
- learning rate: float. Value for the initial learning rate
- model: int. 
Possible values: 0, 2 or 4 (see paper for architecture details).
- batch normalization: leave blank for not using any batch normalization layers, or 1 to use them in all layers of model types 2 or 4.

Other parameters have a default value and can be modified in script.lua: regarding the use of GPUs vs GPUs with CUDNN drivers vs CPUs, the run verbosity and the folder name where to save results.

