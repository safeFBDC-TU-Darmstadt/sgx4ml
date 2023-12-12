# SGX4ML C++

## This Directory

This repository contains the C++ source code used for benchmarking neural network inference in Intel SGX enclaves.
The implementation is based on the Intel SGX DNNL library provided by Intel. As a baseline, we use the same version 
of the library (v1.1.1) without SGX. Both libraries are provided as binary in the [external](external) folder.

To improve the usability of DNNL, we implemented the wrapper class `NeuralNetwork` that provides methods for easy
generation of dense layers, pooling layers, and convolutional layers. Neural network architectures that require other
layer types are not supported.

Training is not supported, however, we implemented a method to import Numpy arrays containing the weights of a
neural network. It is therefore possible to train a network using Pytorch or Tensorflow, export the weights from python
and import the weights into this implementation.

The benchmarks are executed with random weights, biases and data. However, we also tested loading weights from files (as
mentioned above) and inferencing real images from the Fashion MNIST dataset. We achieved similar accuracy as the python
implementation we exported the weights from.

The figures from the paper can be reproduced by running [btw_paper_analysis.py](benchmark-results/BTW-Paper/btw_paper_analysis.py)
according to the guide given under [Using Python for plotting](#using-python-for-plotting).

### Pre-requisites

- Server with support for Intel SGX
- Ubuntu 20.04 with HWE kernel 5.15 or 22.04 with HWE kernel 5.19
- Intel(R) Software Guard Extensions (Intel(R) SGX) SDK for Linux OS version 2.17.1 or newer installed

Might work with similar configurations.

### Run

1. Build the project
   ```shell
   mkdir cmake-build-release && cd cmake-build-release
   CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=Release -DSGX_MODE=PreRelease -DSGX_HW=ON ..
   make -j 16
   ```
   Set `DSGX_HW=SIM` for simulation mode and `-DCMAKE_BUILD_TYPE=Debug` for debugging. The enclave can be debugged with
   `sgx-gdb`.
2. Run
   ```shell
   ./App
   ```
3. Experiment results are saved into file `results.csv`

### Using Python for plotting

1. Create and activate virtual environment
   ```shell
   python3 -m venv ./venv
   source venv/bin/activate
   ```
2. Install required dependencies
   ```shell
   pip install -r requirements.txt
   ```
3. Use Python files in [benchmark-results](benchmark-results) to plot predefined figures or write new figures. Files in
   [benchmark-results](benchmark-results) assume their working directory is their own directory. Example:
   ```shell
   source venv/bin/activate
   cd benchmark-results/BTW-Paper
   python3 btw_paper_analysis.py
   ```
