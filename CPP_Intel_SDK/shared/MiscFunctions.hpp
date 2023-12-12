//
// Created by adrian on 05.10.22.
//

#ifndef SGX_DNNL_MISCFUNCTIONS_HPP
#define SGX_DNNL_MISCFUNCTIONS_HPP

#include <vector>
#include <algorithm>

long argmax(std::vector<float> vec) {
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

std::vector<long> batched_argmax(std::vector<float> vec, long batch_size) {
    long result_size = static_cast<long>(vec.size()) / batch_size;
    std::vector<long> result_indices(batch_size);

    for (long i = 0; i < batch_size; ++i) {
        auto result_start = vec.begin() + (result_size * i);
        auto result_end = result_start + result_size;
        result_indices[i] = std::distance(result_start, std::max_element(result_start, result_end));
    }
    return result_indices;
}

void batched_argmax(std::vector<float> vec, long batch_size, uint16_t *result) {
    long result_size = static_cast<long>(vec.size()) / batch_size;

    for (long i = 0; i < batch_size; ++i) {
        auto result_start = vec.begin() + (result_size * i);
        auto result_end = result_start + result_size;
        result[i] = static_cast<uint16_t>(
                std::distance(result_start, std::max_element(result_start, result_end))
                );
    }
}

#endif //SGX_DNNL_MISCFUNCTIONS_HPP
