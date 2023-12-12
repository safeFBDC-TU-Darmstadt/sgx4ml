// App.cpp : Define the entry point for the console application.
//

#include <iostream>
#include <dnnl.hpp>

#include "sgx_urts.h"
#include "mnist_reader.hpp"
#include "Config.hpp"

constexpr int DEFAULT_THREADS = 16;
constexpr int REPETITIONS = 100;
using NNWeights=const std::vector<std::vector<float>>;
using MNIST=const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t>;

#include "NumpyArrayHelpers.hpp"
#include "UnsafeBenchmarks.hpp"
#include "EnclaveBenchmarks.hpp"

void benchmark_with_data_load(dnnl::engine &engine, dnnl::stream &engine_stream) {

    auto dataset = mnist::read_dataset<std::vector, std::vector, float, uint8_t>("../data");

    std::cout << "Fashion MNIST loaded\n";

    auto mlp_weights = loadVectorsFromArrayFiles({"../data/l1_weights.npy", "../data/l2_weights.npy"});
    auto mlp_biases = loadVectorsFromArrayFiles({"../data/l1_bias.npy", "../data/l2_bias.npy"});

    std::cout << "MLP weights loaded" << std::endl;

    auto cnn_weights = loadVectorsFromArrayFiles({"../data/cnn_conv1_weights.npy",
                                                  "../data/cnn_conv2_weights.npy",
                                                  "../data/cnn_l1_weights.npy",
                                                  "../data/cnn_l2_weights.npy",
                                                  "../data/cnn_l3_weights.npy"});

    auto cnn_biases = loadVectorsFromArrayFiles({"../data/cnn_conv1_bias.npy",
                                                 "../data/cnn_conv2_bias.npy",
                                                 "../data/cnn_l1_bias.npy",
                                                 "../data/cnn_l2_bias.npy",
                                                 "../data/cnn_l3_bias.npy"});

    std::cout << "CNN weights loaded" << std::endl;

    std::cout << "Starting Extended PyTorch Comparison Benchmark with data copy" << std::endl;

    std::ofstream out_file{"results.csv", std::ios::out | std::ios::trunc};

    out_file << "sgx,copy,nn,threads,batch_size,inference_time" << std::endl;

    std::vector<std::string> networks {"mlp", "medium_mlp", "large_mlp", "cnn", "simple_alex", "alex", "vgg16", "vgg19"};
    std::vector<long> batch_sizes {1};
    std::vector<int> thread_counts {16};

    sgx_enclave_id_t eid;
    auto enclave_setup_successful = setup_reduced_enclave(ENCLAVE_NAME, eid);

    if (enclave_setup_successful) {
        //benchmark_pytorch_comparison_SGX_ext(eid, out_file, mlp_weights, mlp_biases, cnn_weights, cnn_biases,
        //                                     networks, batch_sizes, thread_counts);
        enclave_benchmark_including_copy(eid, out_file, std::make_unique<NNWeights>(mlp_weights),
                std::make_unique<NNWeights>(mlp_biases), std::make_unique<NNWeights>(cnn_weights), std::make_unique<NNWeights>(cnn_biases),
                                         networks, batch_sizes, thread_counts);
    }

    sgx_destroy_enclave(eid);

    native_benchmark(engine, engine_stream, out_file, std::make_unique<MNIST>(dataset),
                     std::make_unique<NNWeights>(mlp_weights), std::make_unique<NNWeights>(mlp_biases),
                     std::make_unique<NNWeights>(cnn_weights), std::make_unique<NNWeights>(cnn_biases),
                     networks, batch_sizes, thread_counts);

    out_file.flush();
    out_file.close();
}

void benchmark(dnnl::engine &engine, dnnl::stream &engine_stream) {
    std::cout << "Starting Extended PyTorch Comparison Benchmark with data copy" << std::endl;

    std::ofstream out_file{"results.csv", std::ios::out | std::ios::trunc};

    out_file << "sgx,copy,nn,threads,batch_size,inference_time" << std::endl;

    std::vector<std::string> networks {"mlp", "medium_mlp", "large_mlp", "cnn", "simple_alex", "alex", "vgg16", "vgg19"};
    std::vector<long> batch_sizes {16, 1};
    std::vector<int> thread_counts {16, 8, 4, 2, 1};

    sgx_enclave_id_t eid;
    auto enclave_setup_successful = setup_reduced_enclave(ENCLAVE_NAME, eid);

    if (enclave_setup_successful) {
        enclave_benchmark_including_copy(eid, out_file, networks, batch_sizes, thread_counts);
    }

    sgx_destroy_enclave(eid);

    native_benchmark(engine, engine_stream, out_file, networks, batch_sizes, thread_counts);

    out_file.flush();
    out_file.close();
}

int main(int argc, char *argv[]) {
    auto engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    auto engine_stream = dnnl::stream(engine);

    benchmark(engine, engine_stream);

    return 0;
}

