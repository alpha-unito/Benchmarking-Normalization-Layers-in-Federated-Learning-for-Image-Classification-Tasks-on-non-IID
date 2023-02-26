# Benchmarking-Normalization-Layers-in-Federated-Learning-for-Image-Classification-Tasks-on-non-IID scenarios
This repository contains the code used to execute the experiments described in the paper <i>Benchmarking Normalization Layers in Federated Learning for Image Classification Tasks on non-IID scenarios</i>. Code is written in [PyTorch](https://pytorch.org/) and using the Intel [OpenFL](https://openfl.readthedocs.io/en/latest/index.html) framework.

Extensive experiments have been executed in a centralized and in a federated scenarios (algorithm used: FedAvg), testing a ResNet-18 over two datasets:
- [MNIST](http://yann.lecun.com/exdb/mnist/) 
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

Models have been trained on the [HPC4AI](https://hpc4ai.unito.it/documentation/) cluster at the University of Turin (node: 8 cores per CPU, AMD EPYC-IPBP, 1 NVIDIA A40 GPU));

## Usage

To run the experiment as is, clone [this]([[
](https://github.com/CasellaJr/Benchmarking-Normalization-Layers-in-Federated-Learning-for-Image-Classification-Tasks-on-non-IID)]([
](https://github.com/CasellaJr/Benchmarking-Normalization-Layers-in-Federated-Learning-for-Image-Classification-Tasks-on-non-IID))) repository and use the following:
For the centralized version just run the [MNIST_CENTRALIZED, CIFAR10_CENTRALIZED].ipynb notebooks, selecting the normalization layer you want to use.

For the federated version:
- ```
  1. Install [OpenFL](https://openfl.readthedocs.io/en/latest/index.html).
  2. Put the files containing the non-IID distributions (<i>__init__.py</i> and <i>numpy.py</i>) inside <i>openfl/utilities/data_splitters</i>.
  3. Open a terminal for the director and one for each envoy, after selecting the split you want to use from the shard descriptor, and after creating as many <i>envoy_configX.yaml</i> and <i>start_envoyX.sh</i> you want.
  4. `./start_director.sh`
  5. `./start_envoy.sh`
  6. Run all the cells of the notebook in the workspace, after selecting the normalization layer you want to use.
  7. For reproducibility, do 5 runs changing the variable `myseed` from 0 to 4 and then calculate mean and standard deviation of the best aggregated model.
  ```


## Contributors

Bruno Casella <bruno.casella@unito.it>  
Roberto Esposito <roberto.esposito@unito.it>
Antonio Sciarappa <antonio.sciarappa.ext@leonardo.com>
Carlo Cavazzoni <carlo.cavazzoni@leonardo.com>
Marco Aldinucci <marco.aldinucci@unito.it>  

