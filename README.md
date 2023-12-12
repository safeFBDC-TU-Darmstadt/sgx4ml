# SGX4ML

> For domains with high data privacy and protection demands, such as health care and finance, outsourcing machine 
> learning tasks often requires additional security measures. Trusted Execution Environments like Intel SGX are a 
> powerful tool to achieve this additional security. Until recently, Intel SGX incurred high performance costs, mainly 
> because it was severely limited in terms of available memory and CPUs. With the second generation of SGX, Intel 
> alleviates these problems. Therefore, we revisit previous use cases for ML secured by SGX and show initial results of 
> a performance study for ML workloads on SGXv2. [Abstract of our paper linked below.]

This performance study is especially interesting for the safeFBDC project because TEEs are a viable technology for 
privacy- and sovereignty-preserving data analytics and AI on shared data sets from multiple data owners. The prototypes 
programmed for the performance evaluation can be used as blueprints to port other applications to TEEs and also serve as
a proof of concept.

Our current results indicate that the performance impact of Intel SGX is now very low for compute- and/or 
memory-intensive workloads. We are still investigating I/O-intensive workloads (for example a federated learning 
parameter server). Furthermore, running vanilla applications in SGX enclaves is now simple because of wrappers like 
Gramine (used in this work) and SCONE (not used in this work but evaluated independently).

## Published paper

We published our current results in this paper:

Lutsch, A., Singh, G., Mundt, M., Mogk, R. & Binnig, C., (2023). Benchmarking the Second Generation of Intel SGX for 
Machine Learning Workloads. In: König-Ries, B., Scherzinger, S., Lehner, W. & Vossen, G. (Hrsg.), BTW 2023. Gesellschaft
für Informatik e.V.. DOI: [10.18420/BTW2023-44](https://dx.doi.org/10.18420/BTW2023-44)

### Download

[Paper download via DOI](https://dx.doi.org/10.18420/BTW2023-44)

[Direct paper download.](BTW2023_Paper.pdf)

### BibTeX

```
@inproceedings{mci/Lutsch2023,
author = {Lutsch, Adrian AND Singh, Gagandeep AND Mundt, Martin AND Mogk, Ragnar AND Binnig, Carsten},
title = {Benchmarking the Second Generation of Intel SGX for Machine Learning Workloads},
booktitle = {BTW 2023},
year = {2023},
editor = {König-Ries, Birgitta AND Scherzinger, Stefanie AND Lehner, Wolfgang AND Vossen, Gottfried} ,
doi = { 10.18420/BTW2023-44 },
publisher = {Gesellschaft für Informatik e.V.},
address = {}
}
```

## This Repository

This repository is split into two parts. The directory [CPP Intel SDK](CPP_Intel_SDK) contains the C++ source code used 
for benchmarking neural network inference in Intel SGX enclaves. The implementation is based on the Intel SGX DNNL library 
provided by Intel. The directory [Python Gramine](Python_Gramine) contains Python source code and Gramine configurations 
for the same task. Each directory contains its own README explaining how to use the code.

