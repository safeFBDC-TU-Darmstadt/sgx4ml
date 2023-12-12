//
// Created by adrian on 16.09.22.
//

#ifndef SGX_DNNL_NUMPYARRAY_HPP
#define SGX_DNNL_NUMPYARRAY_HPP

#include <vector>

struct NumpyArray {
    std::vector<float> data;
    std::vector<unsigned long> shape;
    bool order;
};

#endif //SGX_DNNL_NUMPYARRAY_HPP
