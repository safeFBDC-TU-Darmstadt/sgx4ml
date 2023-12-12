//
// Created by adrian on 11.08.22.
//

#include "NeuralNetwork.hpp"
#include "Enclave_t.h"

#include "../shared/BatchedDataset.hpp"
#include "../shared/NeuralNetworkFactories.hpp"

// Global variables that hold the state of the enclave
dnnl::engine engine;
dnnl::stream engine_stream;
std::unique_ptr<NeuralNetwork> nnp;
std::vector<std::vector<float>> dataset;
std::vector<uint8_t> labels;
BatchedDataset batched_dataset;
std::vector<std::vector<float>> result_batches;

extern "C" {

void ecall_setup_dnnl() {
    engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    engine_stream = dnnl::stream(engine);
}

void ecall_create_NN(const char *name, long batch_size, const void *weights_pointer, const void *biases_pointer) {
    if ((std::strcmp(name, "mlp") == 0 || std::strcmp(name, "cnn") == 0)
    && weights_pointer != nullptr && biases_pointer != nullptr) {
        auto weights = *static_cast<const std::vector<std::vector<float>> *>(weights_pointer);
        auto biases = *static_cast<const std::vector<std::vector<float>> *>(biases_pointer);

        nnp = std::make_unique<NeuralNetwork>(create_NN(engine, engine_stream, name, batch_size, weights, biases));
        return;
    }

    nnp = std::make_unique<NeuralNetwork>(create_NN(engine, engine_stream, name, batch_size));
}

void ecall_link_dataset(const void *dataset_p) {
    // Assignment of vectors is implemented as deep copy. Therefore, this operation copies the dataset from the unsafe
    // memory region to the enclave
    dataset = *static_cast<const std::vector<std::vector<float>> *>(dataset_p);
}

void ecall_link_labels(const void *labels_p) {
    // Assignment of vectors is implemented as deep copy. Therefore, this operation copies the labels from the unsafe
    // memory region to the enclave
    labels = *static_cast<const std::vector<uint8_t> *>(labels_p);
}

void ecall_load_dataset_to_enclave(float **dataset_p, size_t image_size, size_t dataset_size) {
    for (float **image_p = dataset_p; image_p != dataset_p + dataset_size; ++image_p) {
        dataset.emplace_back(*image_p, *image_p + image_size);
    }
}

void ecall_load_labels_to_enclave(uint8_t *label_p, size_t label_size) {
    labels.resize(label_size);
    labels.assign(label_p, label_p + label_size);
}

void ecall_create_batched_dataset(size_t batch_size) {
    batched_dataset = BatchedDataset {dataset, labels, batch_size};
}

void ecall_create_random_batched_dataset(size_t batch_size, size_t image_size, size_t batch_count) {
    batched_dataset = BatchedDataset {batch_size, image_size, batch_count};
}

void ecall_run_inference(int *result) {
    *result = 0;

    for (int i = 0; i < dataset.size(); ++i) {
        auto result_vector = nnp->infer(dataset[i]);
        auto result_index = argmax(result_vector);
        auto correct_label = labels[i];

        *result += static_cast<int>(correct_label == result_index);
    }
}

void ecall_batched_inference() {
    result_batches.clear();
    for (auto const &batch: batched_dataset.batched_data) {
        result_batches.emplace_back(nnp->infer(batch));
    }
}

size_t ecall_get_batch_count() {
    return batched_dataset.batch_count;
}

void ecall_batch_inference_prepare() {
    result_batches.clear();
}

void ecall_batch_input(int batch_index) {
    try {
        nnp->writeInput(batched_dataset.batched_data.at(batch_index));
    } catch (std::exception &ex) {
        ocall_print_string(ex.what());
    }
}

void ecall_batch_execute() {
    nnp->execute();
}

void ecall_batch_save_result() {
    result_batches.emplace_back(nnp->readOutput());
}

void ecall_batch_infer(int batch_index) {
    result_batches.emplace_back(nnp->infer(batched_dataset.batched_data.at(batch_index)));
}

void ecall_batch_infer_in_out_vector(float * input_data, size_t input_count, float * result, size_t result_count) {
    nnp->infer(input_data, input_count, result, result_count);
}

void ecall_batch_infer_in_out_classes(float * input_data, size_t input_count, uint16_t * result, size_t batch_size) {
    if (nnp->getBatchSize() != batch_size) {
        throw std::runtime_error("Error in ecall_batch_infer_in_out_long: Output size does not equal buffer size.");
    }
    auto result_vector = nnp->infer(input_data, input_count);
    batched_argmax(result_vector, nnp->getBatchSize(), result);
}

void ecall_check_results(int *result) {
    *result = batched_dataset.check_results(result_batches);
}

void ecall_infer_one() {
    nnp->infer(dataset.at(1));
}

}