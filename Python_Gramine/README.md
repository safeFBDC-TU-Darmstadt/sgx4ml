# SGX4ML Python

## This Directory

This repository contains Python source code and Gramine configurations used for benchmarking neural network inference 
in Intel SGX enclaves. The implementation is based on Python 3.8 and PyTorch 1.13. As a baseline, we use the same 
version we ran the same Python code without Gramine/SGX.

Until now, we have neither tested nor benchmarked training, but we do not expect problems. The benchmarks are executed 
with random weights, biases and data. Except for VGG16 and VGG19 that we load as pretrained networks from 
[PyTorch Hub](https://pytorch.org/hub/).

The results of the benchmarks in this repository did not make it into our short paper mentioned above. We will report
on the results in an upcoming full paper.

### Pre-requisites

- Server with support for Intel SGX
- Ubuntu 20.04 with HWE Kernel 5.15
- Intel(R) Software Guard Extensions (Intel(R) SGX) SDK for Linux* OS version 2.17.1 installed

Might work with similar configurations. We used the upstream kernel SGX driver.

### Installation

1. Install Gramine and prepare signing key: <https://gramine.readthedocs.io/en/stable/quickstart.html>
2. Install PyTorch:
   ```shell
   sudo pip3 install torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   Note that this does not work with virtual environments or similar because of the Gramine manifest. The packages
   must be installed system-wide.
3. Pull repository
4. Optional if you get an error message regarding protobuf:
   ```shell
   sudo pip3 install protobuf==3.12
   ```

### Run Experiments

1. Create models `python3 create_models.py`
2. run `./exp.sh`

Output: File `results`
