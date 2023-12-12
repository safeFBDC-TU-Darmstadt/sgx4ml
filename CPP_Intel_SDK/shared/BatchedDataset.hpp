//
// Created by adrian on 27.09.22.
//

#ifndef SGX_DNNL_BATCHEDDATASET_HPP
#define SGX_DNNL_BATCHEDDATASET_HPP

#include <vector>
#include <random>
#include <algorithm>
#include "MiscFunctions.hpp"

class BatchedDataset {
public:
    size_t batch_size{1};
    size_t image_size{1};
    size_t batch_size_in_floats{1};
    size_t batch_count{0};
    std::vector<std::vector<float>> batched_data;
    std::vector<std::vector<uint8_t>> batched_labels;

    BatchedDataset() = default;

    BatchedDataset(const std::vector<std::vector<float>> &input_data, const std::vector<uint8_t> &labels,
                   size_t batch_size) :
            batch_size{batch_size},
            image_size{input_data[0].size()},
            batch_size_in_floats{batch_size * image_size},
            batch_count{(input_data.size() + batch_size - 1) / batch_size},
            batched_data(batch_count),
            batched_labels(batch_count) {
        for (size_t batch_i = 0; batch_i < batch_count; ++batch_i) {

            auto &current_batch = batched_data[batch_i];
            current_batch.resize(batch_size_in_floats);

            auto &current_label_batch = batched_labels[batch_i];
            current_label_batch.resize(batch_size);

            try {
                for (size_t in_batch_i = 0; in_batch_i < batch_size; ++in_batch_i) {
                    auto current_image_i = batch_i * batch_size + in_batch_i;
                    auto &current_image = input_data.at(current_image_i);
                    std::copy(current_image.begin(),
                              current_image.end(),
                              current_batch.begin() + (in_batch_i * image_size));

                    current_label_batch[in_batch_i] = labels.at(current_image_i);
                }
            } catch (std::out_of_range &out_of_range_error) {
                // Out of range is expected for the last batch
                if (batch_i != batch_count - 1) {
                    //std::cout << "Unexpected out of range error during batching!" << std::endl;
                    throw std::out_of_range("Unexpected out of range error during batching!");
                }
            }
        }
    }

    BatchedDataset(size_t batch_size, size_t image_size, size_t batch_count,
                   uint8_t max_label = std::numeric_limits<uint8_t>::max()) : batch_size{batch_size},
                                                                              image_size{image_size},
                                                                              batch_size_in_floats{
                                                                                batch_size * image_size},
                                                                              batch_count{batch_count},
                                                                              batched_data(batch_count),
                                                                              batched_labels(batch_count) {
        std::random_device rnd_device;
        std::mt19937 mersenne_engine{rnd_device()};  // Generates random integers
        std::normal_distribution<float> normal_distribution;
        std::uniform_int_distribution<uint8_t> uniform_distribution{0, max_label};

        auto normal_float_generator = [&normal_distribution, &mersenne_engine]() {
            return normal_distribution(mersenne_engine);
        };

        auto uniform_int_generator = [&uniform_distribution, &mersenne_engine]() {
            return uniform_distribution(mersenne_engine);
        };

        for (size_t batch_i = 0; batch_i < batch_count; ++batch_i) {
            auto &current_batch = batched_data[batch_i];
            current_batch.resize(batch_size_in_floats);

            auto &current_label_batch = batched_labels[batch_i];
            current_label_batch.resize(batch_size);

            std::generate(current_batch.begin(), current_batch.end(), normal_float_generator);
            std::generate(current_label_batch.begin(), current_label_batch.end(), uniform_int_generator);
        }
    }

    int check_results(const std::vector<std::vector<float>> &results) const {
        int result = 0;

        for (size_t batch_i = 0; batch_i < batch_count; ++batch_i) {
            auto result_indices = batched_argmax(results[batch_i], batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                result += (result_indices[i] == batched_labels[batch_i][i]);
            }
        }

        return result;
    }

    int check_results(const std::vector<std::vector<uint16_t>> &results) const {
        int result = 0;

        for (size_t batch_i = 0; batch_i < batch_count; ++batch_i) {
            for (size_t i = 0; i < batch_size; ++i) {
                result += (results[batch_i][i] == batched_labels[batch_i][i]);
            }
        }

        return result;
    }

    int check_batch(const std::vector<float> &results, size_t batch_i) const {
        int result = 0;

        auto result_indices = batched_argmax(results, batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            result += (result_indices[i] == batched_labels[batch_i][i]);
        }

        return result;
    }
};

#endif //SGX_DNNL_BATCHEDDATASET_HPP
