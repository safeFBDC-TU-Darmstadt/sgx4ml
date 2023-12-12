//
// Created by adrian on 15.09.22.
//

#ifndef SGX_DNNL_ENCLAVEBENCHMARKS_HPP
#define SGX_DNNL_ENCLAVEBENCHMARKS_HPP

#include "Enclave_u.h"
#include "Timers.hpp"
#include "SGXErrors.hpp"
#include "ExperimentHelpers.hpp"

bool set_number_of_threads(sgx_enclave_id_t eid, size_t num_threads) {
    sgx_status_t ret {};
    ret = ecall_set_threads(eid, num_threads);
    if (ret == SGX_SUCCESS) {
        std::cout << "Number of threads: " + std::to_string(num_threads) << std::endl;
    } else {
        std::cout << "Setting number of threads failed!" << std::endl;
        print_error_message(ret);
        return false;
    }
    return true;
}

/**
 * Creates the enclave, starts DNNL inside the enclave, sets the number of threads, and loads the dataset into the
 * enclave.
 * @param enclave_name File name of the enclave to load
 * @param eid OUTPUT: id of the created enclave
 * @param dataset dataset to load into the enclave
 * @return
 */
bool setup_enclave(const char * enclave_name, sgx_enclave_id_t &eid,
                   const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> &dataset) {
    sgx_status_t ret {};

    // load the enclave
    // Debug: set the 2nd parameter to 1 which indicates the enclave are launched in debug mode
    {
        Timers::BlockTimer bt{"Enclave creation", std::cout};
        ret = sgx_create_enclave(enclave_name, SGX_DEBUG_FLAG, nullptr, nullptr, &eid, nullptr);
    }
    if (ret == SGX_SUCCESS) {
        std::cout << "Enclave created." << std::endl;
    } else {
        std::cout << "Enclave creation failed!" << std::endl;
        print_error_message(ret);
        return false;
    }

    {
        Timers::BlockTimer bt{"NOOP", std::cout};
        ecall_noop(eid);
    }

    {
        Timers::BlockTimer bt{"NOOP 2", std::cout};
        ecall_noop(eid);
    }

    {
        Timers::BlockTimer bt{"NOOP 3", std::cout};
        ecall_noop(eid);
    }

    {
        Timers::BlockTimer bt{"DNNL Setup", std::cout};
        ret = ecall_setup_dnnl(eid);
    }

    if (ret == SGX_SUCCESS) {
        std::cout << "DNNL setup done." << std::endl;
    } else {
        std::cout << "Setup DNNL failed!" << std::endl;
        print_error_message(ret);
        return false;
    }

    if (!set_number_of_threads(eid, DEFAULT_THREADS)) {
        return false;
    }

    {
        Timers::BlockTimer bt{"Dataset copy to enclave", std::cout};
        ret = ecall_link_dataset(eid, static_cast<const void *>(&dataset.training_images));
    }
    if (ret != SGX_SUCCESS) {
        std::cout << "Linking the dataset into the enclave failed!" << std::endl;
        print_error_message(ret);
        return false;
    }

    {
        Timers::BlockTimer bt{"Labels copy to enclave", std::cout};
        ret = ecall_link_labels(eid, static_cast<const void *>(&dataset.training_labels));
    }
    if (ret != SGX_SUCCESS) {
        std::cout << "Linking the labels into the enclave failed!" << std::endl;
        print_error_message(ret);
        return false;
    }

    std::cout << "Dataset and labels loaded." << std::endl;
    return true;
}

/**
 * Creates the enclave, starts DNNL inside the enclave, and sets the number of threads
 * @param enclave_name File name of the enclave to load
 * @param eid OUTPUT: id of the created enclave
 * @return
 */
bool setup_reduced_enclave(const char * enclave_name, sgx_enclave_id_t &eid) {
    sgx_status_t ret {};

    // load the enclave
    // Debug: set the 2nd parameter to 1 which indicates the enclave are launched in debug mode
    {
        Timers::BlockTimer bt{"Enclave creation", std::cout};
        ret = sgx_create_enclave(enclave_name, SGX_DEBUG_FLAG, nullptr, nullptr, &eid, nullptr);
    }
    if (ret == SGX_SUCCESS) {
        std::cout << "Enclave created." << std::endl;
    } else {
        std::cout << "Enclave creation failed!" << std::endl;
        print_error_message(ret);
        return false;
    }

    {
        Timers::BlockTimer bt{"NOOP", std::cout};
        ecall_noop(eid);
    }

    {
        Timers::BlockTimer bt{"NOOP 2", std::cout};
        ecall_noop(eid);
    }

    {
        Timers::BlockTimer bt{"NOOP 3", std::cout};
        ecall_noop(eid);
    }

    {
        Timers::BlockTimer bt{"NOOP 4", std::cout};
        ecall_noop(eid);
    }

    {
        Timers::BlockTimer bt{"DNNL Setup", std::cout};
        ret = ecall_setup_dnnl(eid);
    }

    if (ret == SGX_SUCCESS) {
        std::cout << "DNNL setup done." << std::endl;
    } else {
        std::cout << "Setup DNNL failed!" << std::endl;
        print_error_message(ret);
        return false;
    }

    if (!set_number_of_threads(eid, DEFAULT_THREADS)) {
        return false;
    }

    return true;
}

bool setup_mlp(sgx_enclave_id_t &eid,
               const std::vector<std::vector<float>> &mlp_weights,
               const std::vector<std::vector<float>> &mlp_biases,
               long batch_size = 1) {
    sgx_status_t ret {};
    ret = ecall_create_NN(eid, "mlp", batch_size, &mlp_weights, &mlp_biases);
    if (ret != SGX_SUCCESS) {
        std::cout << "Creating the MLP with the given data failed!" << std::endl;
        print_error_message(ret);
        return false;
    }
    return true;
}

bool setup_cnn(sgx_enclave_id_t &eid, const std::vector<std::vector<float>> &cnn_weights,
               const std::vector<std::vector<float>> &cnn_biases, long batch_size = 1) {
    sgx_status_t ret {};
    ret = ecall_create_NN(eid, "cnn", batch_size, &cnn_weights, &cnn_biases);
    if (ret != SGX_SUCCESS) {
        std::cout << "Creating the CNN with the given data pointers failed!" << std::endl;
        print_error_message(ret);
        return false;
    }
    return true;
}

void benchmark_enclave(sgx_enclave_id_t eid) {
    sgx_status_t ret {};

    std::cout << "Starting inference!" << std::endl;

    int correct_inferences = 0;
    {
        Timers::BlockTimer bt{"Enclave inference", std::cout};
        ret = ecall_run_inference(eid, &correct_inferences);
    }
    if (ret != SGX_SUCCESS) {
        std::cout << "Inference inside the enclave failed!" << std::endl;
        print_error_message(ret);
        return;
    }
    std::cout << "Inference done.\n";
    std::cout << "Number of correctly classified images (trained MLP): " << correct_inferences << "\n";

    {
        Timers::BlockTimer bt{"Safe inference of one image", std::cout};
        ecall_infer_one(eid);
    }
}

void benchmark_enclave_threads(sgx_enclave_id_t eid, const char *out_file_name) {
    sgx_status_t ret {};

    std::ofstream out_file {out_file_name, std::ios::out | std::ios::trunc};
    //std::ostream &out_file = std::cout;

    out_file << "num_threads,time\n";

    for (int num_threads = 1; num_threads <= 16; ++num_threads) {
        std::cout << num_threads << std::endl;
        ret = ecall_set_threads(eid, num_threads);
        if (ret != SGX_SUCCESS) {
            std::cout << "Failed to set numer of threads!" << std::endl;
            print_error_message(ret);
            return;
        }

        int correct_inferences = 0;
        {
            Timers::BlockTimer bt{std::to_string(num_threads), ",", false, out_file};
            ret = ecall_run_inference(eid, &correct_inferences);
        }
        out_file.flush();
        if (ret != SGX_SUCCESS) {
            std::cout << "Inference inside the enclave failed!" << std::endl;
            print_error_message(ret);
            return;
        }
    }
    out_file.flush();
    out_file.close();
    std::cout << "Benchmark done.\n";
}

void benchmark_enclave_batches(sgx_enclave_id_t eid, const char *out_file_name,
                               const std::vector<std::vector<float>> &mlp_weights,
                               const std::vector<std::vector<float>> &mlp_biases) {
    sgx_status_t ret {};

    ret = ecall_set_threads(eid, DEFAULT_THREADS);
    if (ret != SGX_SUCCESS) {
        std::cout << "Failed to set numer of threads!" << std::endl;
        print_error_message(ret);
        return;
    }

    std::ofstream out_file {out_file_name, std::ios::out | std::ios::trunc};
    //std::ostream &out_file = std::cout;

    out_file << "batch_size,time\n";

    for (size_t batch_size_exponent = 4; batch_size_exponent <= 14; ++batch_size_exponent) {
        size_t batch_size = 1L << batch_size_exponent;

        std::cout << batch_size << std::endl;

        {
            Timers::BlockTimer bt{"batching the dataset", std::cout};
            ret = ecall_create_batched_dataset(eid, batch_size);
        }
        if (ret != SGX_SUCCESS) {
            std::cout << "Dataset batching failed!" << std::endl;
            print_error_message(ret);
            return;
        }

        // std::cout << "Batched dataset created!" << std::endl;

        auto mlp_setup_successful = setup_mlp(eid, mlp_weights, mlp_biases, batch_size);
        if (!mlp_setup_successful) {
            std::cout << "Creating MLP for batch Benchmark failed!" << std::endl;
            print_error_message(ret);
            return;
        }

        // std::cout << "MLP for batch size created!" << std::endl;

        {
            Timers::BlockTimer bt{std::to_string(batch_size), ",", false, out_file};
            ret = ecall_batched_inference(eid);
        }

        if (ret != SGX_SUCCESS) {
            std::cout << "Batched inference failed!" << std::endl;
            print_error_message(ret);
            return;
        }

        // std::cout << "Batched inference done!" << std::endl;

        int correct_matches = 0;
        ret = ecall_check_results(eid, &correct_matches);
        if (ret != SGX_SUCCESS) {
            std::cout << "Checking batched results failed!" << std::endl;
            print_error_message(ret);
            return;
        }
        std::cout << correct_matches << std::endl;

    }
    out_file.flush();
    out_file.close();
    std::cout << "Benchmark done.\n";
}

void benchmark_enclave_batches_split(sgx_enclave_id_t eid, const char *out_file_name,
                                     const std::vector<std::vector<float>> &mlp_weights,
                                     const std::vector<std::vector<float>> &mlp_biases) {
    sgx_status_t ret {};

    ret = ecall_set_threads(eid, DEFAULT_THREADS);
    if (ret != SGX_SUCCESS) {
        std::cout << "Failed to set numer of threads!" << std::endl;
        print_error_message(ret);
        return;
    }

    std::ofstream out_file {out_file_name, std::ios::out | std::ios::trunc};
    //std::ostream &out_file = std::cout;

    out_file << "batch_size,batch_time,copy_time,exec_time,back_time,total_time\n";

    for (size_t batch_size_exponent = 0; batch_size_exponent <= 14; ++batch_size_exponent) {
        size_t batch_size = 1L << batch_size_exponent;

        std::cout << batch_size << std::endl;


        Timers::StopTimer t {};
        ret = ecall_create_batched_dataset(eid, batch_size);
        t.stop();

        auto batching_time = t.duration();

        if (ret != SGX_SUCCESS) {
            std::cout << "Dataset batching failed!" << std::endl;
            print_error_message(ret);
            return;
        }

        // std::cout << "Batched dataset created!" << std::endl;

        auto mlp_setup_successful = setup_mlp(eid, mlp_weights, mlp_biases, batch_size);
        if (!mlp_setup_successful) {
            std::cout << "Creating MLP for batch Benchmark failed!" << std::endl;
            print_error_message(ret);
            return;
        }

        // std::cout << "MLP for batch size created!" << std::endl;

        size_t batch_count = 0;
        ecall_get_batch_count(eid, &batch_count);

        long total_copy_duration = 0;
        long total_execution_duration = 0;
        long total_result_copy_duration = 0;

        ecall_batch_inference_prepare(eid);

        Timers::StopTimer total_timer {};
        for (int batch_index = 0; batch_index < batch_count; ++batch_index) {
            Timers::StopTimer timer {};
            ecall_batch_input(eid, batch_index);
            timer.stop();
            total_copy_duration += timer.duration();
            timer.reset();
            ecall_batch_execute(eid);
            timer.stop();
            total_execution_duration += timer.duration();
            timer.reset();
            ecall_batch_save_result(eid);
            timer.stop();
            total_result_copy_duration += timer.duration();
        }
        total_timer.stop();

        auto avg_copy_duration = total_copy_duration / batch_count;
        auto avg_execution_duration = total_execution_duration / batch_count;
        auto avg_result_copy_duration = total_result_copy_duration / batch_count;

        std::cout << "Batched inference done!\n"
                  << "Batching time: " << batching_time << "\n"
                  << "AVG copy time: " << avg_copy_duration << "\n"
                  << "AVG execution time: " << avg_execution_duration << "\n"
                  << "AVG result copy time: " << avg_result_copy_duration << "\n"
                  << "Total time: " << total_timer.duration() << std::endl;

        out_file << batch_size << ","
                 << batching_time << ","
                 << avg_copy_duration << ","
                 << avg_execution_duration << ","
                 << avg_result_copy_duration << ","
                 << total_timer.duration() << std::endl;

        int correct_matches = 0;
        ret = ecall_check_results(eid, &correct_matches);
        if (ret != SGX_SUCCESS) {
            std::cout << "Checking batched results failed!" << std::endl;
            print_error_message(ret);
            return;
        }
        std::cout << correct_matches << std::endl;

    }
    out_file.flush();
    out_file.close();
    std::cout << "Benchmark done.\n";
}


void benchmark_pytorch_comparison_SGX_ext(sgx_enclave_id_t eid, std::ofstream &out_file,
                                          const std::vector<std::vector<float>> &mlp_weights,
                                          const std::vector<std::vector<float>> &mlp_biases,
                                          const std::vector<std::vector<float>> &cnn_weights,
                                          const std::vector<std::vector<float>> &cnn_biases,
                                          std::vector<std::string> &networks,
                                          std::vector<long> &batch_sizes,
                                          std::vector<int> &thread_counts) {
    std::cout << "Running SGX extended benchmark" << std::endl;

    Timers::StopTimer timer {};

    sgx_status_t ret {};

    for (const auto& network : networks) {
        std::cout << "Network: " << network << std::endl;
        for (auto batch_size : batch_sizes) {
            std::cout << "Batch Size: " << batch_size << std::endl;

            if (network == "mlp") {
                ret = ecall_create_NN(eid, network.c_str(), batch_size, &mlp_weights, &mlp_biases);
            } else if (network == "cnn") {
                ret = ecall_create_NN(eid, network.c_str(), batch_size, &cnn_weights, &cnn_biases);
            } else {
                ret = ecall_create_NN(eid, network.c_str(), batch_size, nullptr, nullptr);
            }
            if (ret != SGX_SUCCESS) {
                print_error_message(ret);
                std::cout << "Creating NN failed, stopping benchmark!" << std::endl;
                return;
            }

            BatchedDataset batched_dataset;
            if (std::find(SMALL_INPUT_NETWORKS.begin(), SMALL_INPUT_NETWORKS.end(), network) != SMALL_INPUT_NETWORKS.end()) {
                batched_dataset = BatchedDataset{static_cast<size_t>(batch_size), 28 * 28, REPETITIONS};
            } else if (std::find(BIG_INPUT_NETWORKS.begin(), BIG_INPUT_NETWORKS.end(), network) != BIG_INPUT_NETWORKS.end()) {
                batched_dataset = BatchedDataset {static_cast<size_t>(batch_size), 3 * 224 * 224, REPETITIONS};
            } else {
                std::cout << "Creating batched dataset failed, stopping benchmark!" << std::endl;
                return;
            }

            for (auto thread_count : thread_counts) {
                ret = ecall_set_threads(eid, thread_count);
                std::cout << "Number of threads: " << thread_count << std::endl;
                if (ret != SGX_SUCCESS) {
                    std::cout << "Setting number of threads failed, stopping benchmark!" << std::endl;
                    return;
                }

                std::vector<long> inference_times{};
                inference_times.reserve(REPETITIONS);

                ret = ecall_batch_inference_prepare(eid);
                if (ret != SGX_SUCCESS) {
                    std::cout << "Inference preparation failed, stopping benchmark!" << std::endl;
                    return;
                }
                for (int test_index = 0; test_index < REPETITIONS; ++test_index) {
                    //std::cout << test_index << std::endl;
                    timer.reset();
                    ret = ecall_batch_infer(eid, test_index);
                    timer.stop();
                    if (ret != SGX_SUCCESS) {
                        std::cout << "Inference failed, stopping benchmark!" << std::endl;
                        print_error_message(ret);
                        return;
                    }
                    inference_times.push_back(timer.duration());
                }

                calculate_and_save_results_copy(out_file, network, thread_count, batch_size, inference_times, true, false);

            }
        }
    }

}

void enclave_benchmark_including_copy(sgx_enclave_id_t eid, std::ofstream& out_file,
                                      std::unique_ptr<const std::vector<std::vector<float>>> mlp_weights,
                                      std::unique_ptr<const std::vector<std::vector<float>>> mlp_biases,
                                      std::unique_ptr<const std::vector<std::vector<float>>> cnn_weights,
                                      std::unique_ptr<const std::vector<std::vector<float>>> cnn_biases,
                                      const std::vector<std::string> &networks,
                                      const std::vector<long> &batch_sizes,
                                      const std::vector<int> &thread_counts) {
    std::cout << "Running SGX extended benchmark with data copy on call" << std::endl;

    Timers::StopTimer timer {};

    sgx_status_t ret {};

    for (const auto& network : networks) {
        std::cout << "Network: " << network << std::endl;
        for (auto batch_size : batch_sizes) {
            std::cout << "Batch Size: " << batch_size << std::endl;

            if (network == "mlp") {
                ret = ecall_create_NN(eid, network.c_str(), batch_size, mlp_weights.get(), mlp_biases.get());
            } else if (network == "cnn") {
                ret = ecall_create_NN(eid, network.c_str(), batch_size, cnn_weights.get(), cnn_biases.get());
            } else {
                ret = ecall_create_NN(eid, network.c_str(), batch_size, nullptr, nullptr);
            }
            if (ret != SGX_SUCCESS) {
                print_error_message(ret);
                std::cout << "Creating NN failed, stopping benchmark!" << std::endl;
                return;
            }

            BatchedDataset batched_dataset;
            if (std::find(SMALL_INPUT_NETWORKS.begin(), SMALL_INPUT_NETWORKS.end(), network) != SMALL_INPUT_NETWORKS.end()) {
                batched_dataset = BatchedDataset{static_cast<size_t>(batch_size), 28 * 28, REPETITIONS};
            } else if (std::find(BIG_INPUT_NETWORKS.begin(), BIG_INPUT_NETWORKS.end(), network) != BIG_INPUT_NETWORKS.end()) {
                batched_dataset = BatchedDataset {static_cast<size_t>(batch_size), 3 * 224 * 224, REPETITIONS};
            } else {
                std::cout << "Creating batched dataset failed, stopping benchmark!" << std::endl;
                return;
            }

            for (auto thread_count : thread_counts) {
                ret = ecall_set_threads(eid, thread_count);
                std::cout << "Number of threads: " << thread_count << std::endl;
                if (ret != SGX_SUCCESS) {
                    std::cout << "Setting number of threads failed, stopping benchmark!" << std::endl;
                    return;
                }

                std::vector<long> inference_times{};
                inference_times.reserve(REPETITIONS);
                std::vector<std::vector<uint16_t>> results(REPETITIONS);

                for (int test_index = 0; test_index < REPETITIONS; ++test_index) {
                    //std::cout << test_index << std::endl;
                    auto &result_vector = results.at(test_index);
                    result_vector.resize(batch_size);
                    timer.reset();
                    ret = ecall_batch_infer_in_out_classes(eid,
                                                           batched_dataset.batched_data[test_index].data(),
                                                           batched_dataset.batch_size_in_floats,
                                                           result_vector.data(),
                                                           batch_size);
                    timer.stop();
                    if (ret != SGX_SUCCESS) {
                        std::cout << "Inference failed, stopping benchmark!" << std::endl;
                        print_error_message(ret);
                        return;
                    }
                    inference_times.push_back(timer.duration());
                }

                calculate_and_save_results_copy(out_file, network, thread_count, batch_size, inference_times, true, true);

            }
        }
    }

}

void enclave_benchmark_including_copy(sgx_enclave_id_t eid, std::ofstream& out_file,
                                      const std::vector<std::string> &networks,
                                      const std::vector<long> &batch_sizes,
                                      const std::vector<int> &thread_counts) {
    enclave_benchmark_including_copy(eid, out_file, nullptr, nullptr, nullptr, nullptr, networks, batch_sizes, thread_counts);
}

#endif //SGX_DNNL_ENCLAVEBENCHMARKS_HPP
