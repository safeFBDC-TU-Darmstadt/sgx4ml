//
// Created by adrian on 15.09.22.
//

#ifndef SGX_DNNL_NUMPYARRAYHELPERS_HPP
#define SGX_DNNL_NUMPYARRAYHELPERS_HPP

#include <numeric>

#include "npy.hpp"
#include "../shared/NumpyArray.hpp"

NumpyArray loadNumpyArray(const std::string &filename) {
    NumpyArray array;
    npy::LoadArrayFromNumpy(filename, array.shape, array.order, array.data);
    return array;
}

std::vector<NumpyArray> loadNumpyArrays(const std::vector<std::string> &filenames) {
    std::vector<NumpyArray> layers{};
    for (const auto &filename: filenames) {
        layers.emplace_back(loadNumpyArray(filename));
    }
    return layers;
}

std::vector<float> loadVector(const std::string &filename) {
    std::vector<float> data{};
    std::vector<unsigned long> shape{};
    bool order = false;
    npy::LoadArrayFromNumpy(filename, shape, order, data);
    return data;
}

std::vector<std::vector<float>> loadVectorsFromArrayFiles(const std::vector<std::string> &filenames) {
    std::vector<std::vector<float>> vectors{};
    for (const auto &filename: filenames) {
        vectors.emplace_back(loadVector(filename));
    }
    return vectors;
}

void analyze_vector(std::vector<float> & vec) {
    auto sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    auto avg = sum / vec.size();
    auto min = *std::min_element(vec.begin(), vec.end());
    auto max = *std::max_element(vec.begin(),vec.end());

    std::cout << vec.size() << "\n"
              << avg << "\n"
              << min << "\n"
              << max << "\n";
}

#endif //SGX_DNNL_NUMPYARRAYHELPERS_HPP
