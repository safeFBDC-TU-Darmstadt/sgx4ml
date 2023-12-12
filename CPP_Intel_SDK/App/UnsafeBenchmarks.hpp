//
// Created by adrian on 15.09.22.
//

#ifndef SGX_DNNL_UNSAFEBENCHMARKS_HPP
#define SGX_DNNL_UNSAFEBENCHMARKS_HPP

#include <random>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include "../../shared/NeuralNetworkFactories.hpp"
#include "Timers.hpp"
#include "ExperimentHelpers.hpp"
#include "../shared/BatchedDataset.hpp"
#include "../shared/MiscFunctions.hpp"


void benchmark_unsafe(dnnl::engine engine, dnnl::stream engine_stream,
                      const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> &dataset,
                      const std::vector<std::vector<float>> &mlp_weights,
                      const std::vector<std::vector<float>> &mlp_biases) {

    NeuralNetwork mlp = create_MLP(engine, engine_stream, mlp_weights, mlp_biases);

    int correct_inferences = 0;
    {
        Timers::BlockTimer bt{"Unsafe inference", std::cout};
        for (int i = 0; i < dataset.training_images.size(); ++i) {
            auto result_vector = mlp.infer(dataset.training_images[i]);
            auto result_index = argmax(result_vector);
            auto correct_label = dataset.training_labels[i];

            correct_inferences += static_cast<int>(correct_label == result_index);
        }
    }
    std::cout << "Inference done.\n";
    std::cout << "Number of correctly classified images (trained MLP): " << correct_inferences << "\n";
    {
        Timers::BlockTimer bt{"Unsafe inference of one image", std::cout};
        mlp.infer(dataset.training_images[1]);
    }
}

void benchmark_unsafe_threads(dnnl::engine engine, dnnl::stream engine_stream,
                              const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> &dataset,
                              const std::vector<std::vector<float>> &mlp_weights,
                              const std::vector<std::vector<float>> &mlp_biases) {

    NeuralNetwork mlp = create_MLP(engine, engine_stream, mlp_weights, mlp_biases);

    std::ofstream out_file{"unsafe_threads_results.csv", std::ios::out | std::ios::trunc};

    out_file << "num_threads,time\n";

    for (int num_threads = 1; num_threads <= 64; ++num_threads) {
        omp_set_num_threads(num_threads);
        std::cout << num_threads << std::endl;
        std::vector<std::vector<float>> result_batches{};
        //int correct_inferences = 0;
        {
            Timers::BlockTimer bt{std::to_string(num_threads), ",", false, out_file};
            for (const auto &training_image: dataset.training_images) {
                result_batches.push_back(mlp.infer(training_image));
            }
        }
    }
    out_file.flush();
    out_file.close();
    std::cout << "Benchmark done.\n";
}

void benchmark_unsafe_threads_batched(dnnl::engine engine, dnnl::stream engine_stream,
                                      long batch_size,
                                      const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> &dataset,
                                      const std::vector<std::vector<float>> &mlp_weights,
                                      const std::vector<std::vector<float>> &mlp_biases) {

    BatchedDataset batched_dataset{dataset.training_images, dataset.training_labels, static_cast<size_t>(batch_size)};
    NeuralNetwork mlp = create_MLP(engine, engine_stream, mlp_weights, mlp_biases, batch_size);

    std::ofstream out_file{"unsafe_threads_results_batched_" + std::to_string(batch_size) + ".csv",
                           std::ios::out | std::ios::trunc};

    out_file << "num_threads,time\n";

    for (int num_threads = 1; num_threads <= 64; ++num_threads) {
        omp_set_num_threads(num_threads);
        std::cout << num_threads << std::endl;
        std::vector<std::vector<float>> result_batches{};
        {
            Timers::BlockTimer bt{std::to_string(num_threads), ",", false, out_file};
            for (auto &batch: batched_dataset.batched_data) {
                result_batches.push_back(mlp.infer(batch));
            }
        }

        int correct_matches = 0;
        for (size_t batch_i = 0; batch_i < batched_dataset.batch_count; ++batch_i) {
            auto result_indices = batched_argmax(result_batches[batch_i], batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                correct_matches += (result_indices[i] == batched_dataset.batched_labels[batch_i][i]);
            }
        }

        std::cout << correct_matches << std::endl;

    }
    out_file.flush();
    out_file.close();
    std::cout << "Benchmark done.\n";
}

void benchmark_unsafe_batch_size(dnnl::engine engine, dnnl::stream engine_stream,
                                 const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> &dataset,
                                 const std::vector<std::vector<float>> &mlp_weights,
                                 const std::vector<std::vector<float>> &mlp_biases) {

    std::ofstream out_file{"unsafe_batches_results_16.csv", std::ios::out | std::ios::trunc};

    out_file << "batch_size,time\n";

    for (size_t batch_size_exponent = 0; batch_size_exponent <= 14; ++batch_size_exponent) {
        long batch_size = 1L << batch_size_exponent;

        Timers::StopTimer stop_timer{};
        BatchedDataset batched_dataset{dataset.training_images, dataset.training_labels,
                                       static_cast<size_t>(batch_size)};
        stop_timer.stop();
        std::cout << stop_timer.duration() << " µs\n";

        NeuralNetwork mlp = create_MLP(engine, engine_stream, mlp_weights, mlp_biases, batch_size);

        std::cout << batch_size << std::endl;

        std::vector<std::vector<float>> result_batches{};
        {
            Timers::BlockTimer bt{std::to_string(batch_size), ",", false, out_file};
            for (auto const &batch: batched_dataset.batched_data) {
                result_batches.push_back(mlp.infer(batch));
            }
        }

        int correct_matches = batched_dataset.check_results(result_batches);

        std::cout << correct_matches << std::endl;

    }
    out_file.flush();
    out_file.close();
    std::cout << "Benchmark done.\n";

}

void benchmark_unsafe_batch_split(dnnl::engine engine, dnnl::stream engine_stream,
                                  const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> &dataset,
                                  const std::vector<std::vector<float>> &mlp_weights,
                                  const std::vector<std::vector<float>> &mlp_biases) {

    std::ofstream out_file{"unsafe_batches_split_16_pin.csv", std::ios::out | std::ios::trunc};

    out_file << "batch_size,batch_time,copy_time,exec_time,back_time,total_time\n";

    for (size_t batch_size_exponent = 0; batch_size_exponent <= 14; ++batch_size_exponent) {
        long batch_size = 1L << batch_size_exponent;

        Timers::StopTimer stop_timer{};
        BatchedDataset batched_dataset{dataset.training_images, dataset.training_labels,
                                       static_cast<size_t>(batch_size)};
        stop_timer.stop();
        auto batching_time = stop_timer.duration();

        NeuralNetwork mlp = create_MLP(engine, engine_stream, mlp_weights, mlp_biases, batch_size);

        std::cout << batch_size << std::endl;

        long total_copy_duration = 0;
        long total_execution_duration = 0;
        long total_result_copy_duration = 0;

        std::vector<std::vector<float>> result_batches{};
        result_batches.reserve(batched_dataset.batch_count);

        Timers::StopTimer total_timer{};
        for (auto const &batch: batched_dataset.batched_data) {
            Timers::StopTimer timer{};
            mlp.writeInput(batch);
            timer.stop();
            total_copy_duration += timer.duration();
            timer.reset();
            mlp.execute();
            timer.stop();
            total_execution_duration += timer.duration();
            timer.reset();
            result_batches.emplace_back(mlp.readOutput());
            timer.stop();
            total_result_copy_duration += timer.duration();
        }
        total_timer.stop();

        auto avg_copy_duration = total_copy_duration / batched_dataset.batch_count;
        auto avg_execution_duration = total_execution_duration / batched_dataset.batch_count;
        auto avg_result_copy_duration = total_result_copy_duration / batched_dataset.batch_count;

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

        int correct_matches = batched_dataset.check_results(result_batches);

        std::cout << correct_matches << std::endl;

    }
    out_file.flush();
    out_file.close();
    std::cout << "Benchmark done.\n";
}

void benchmark_unsafe_cnn_batch_split(dnnl::engine engine, dnnl::stream engine_stream,
                                      const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> &dataset,
                                      const std::vector<std::vector<float>> &cnn_weights,
                                      const std::vector<std::vector<float>> &cnn_biases) {

    std::ofstream out_file{"unsafe_batches_split_16_pin.csv", std::ios::out | std::ios::trunc};

    out_file << "batch_size,batch_time,copy_time,exec_time,back_time,total_time\n";

    for (size_t batch_size_exponent = 0; batch_size_exponent <= 14; ++batch_size_exponent) {
        long batch_size = 1L << batch_size_exponent;

        Timers::StopTimer stop_timer{};
        BatchedDataset batched_dataset{dataset.training_images, dataset.training_labels,
                                       static_cast<size_t>(batch_size)};
        stop_timer.stop();
        auto batching_time = stop_timer.duration();
        NeuralNetwork cnn = create_CNN(engine, engine_stream, cnn_weights, cnn_biases, batch_size);

        std::cout << "CNN created.\n";

        std::cout << batch_size << std::endl;

        long total_copy_duration = 0;
        long total_execution_duration = 0;
        long total_result_copy_duration = 0;

        std::vector<std::vector<float>> result_batches{};
        result_batches.reserve(batched_dataset.batch_count);

        Timers::StopTimer total_timer{};
        for (auto const &batch: batched_dataset.batched_data) {
            Timers::StopTimer timer{};
            cnn.writeInput(batch);
            timer.stop();
            total_copy_duration += timer.duration();
            timer.reset();
            cnn.execute();
            timer.stop();
            total_execution_duration += timer.duration();
            timer.reset();
            result_batches.emplace_back(cnn.readOutput());
            timer.stop();
            total_result_copy_duration += timer.duration();
        }
        total_timer.stop();

        auto avg_copy_duration = total_copy_duration / batched_dataset.batch_count;
        auto avg_execution_duration = total_execution_duration / batched_dataset.batch_count;
        auto avg_result_copy_duration = total_result_copy_duration / batched_dataset.batch_count;

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

        int correct_matches = batched_dataset.check_results(result_batches);

        std::cout << correct_matches << std::endl;

    }
    out_file.flush();
    out_file.close();
    std::cout << "Benchmark done.\n";

}

void benchmark_single_split(dnnl::engine engine, dnnl::stream engine_stream,
                            const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> &dataset,
                            const std::vector<std::vector<float>> &mlp_weights,
                            const std::vector<std::vector<float>> &mlp_biases) {

    std::ofstream out_file{"unsafe_single_split_16.csv", std::ios::out | std::ios::trunc};

    out_file << "copy_time,exec_time,back_time,total_time\n";

    NeuralNetwork mlp = create_MLP(engine, engine_stream, mlp_weights, mlp_biases);

    auto image_count = dataset.training_images.size();
    std::vector<std::vector<float>> results{};
    results.reserve(image_count);
    int correct_inferences = 0;

    long total_copy_duration = 0;
    long total_execution_duration = 0;
    long total_result_copy_duration = 0;
    std::vector<long> execution_duration_vector{};

    Timers::StopTimer total_timer{};
    for (auto const &image: dataset.training_images) {
        Timers::StopTimer timer{};
        mlp.writeInput(image);
        timer.stop();
        total_copy_duration += timer.duration();
        timer.reset();
        mlp.execute();
        timer.stop();
        total_execution_duration += timer.duration();
        execution_duration_vector.emplace_back(timer.duration());
        timer.reset();
        results.emplace_back(mlp.readOutput());
        timer.stop();
        total_result_copy_duration += timer.duration();
        auto result_index = argmax(results[results.size() - 1]);
        auto correct_label = dataset.training_labels[results.size() - 1];

        correct_inferences += static_cast<int>(correct_label == result_index);
    }
    total_timer.stop();

    auto avg_copy_duration = total_copy_duration / image_count;
    auto avg_execution_duration = total_execution_duration / image_count;
    auto avg_result_copy_duration = total_result_copy_duration / image_count;

    std::cout << "Batched inference done!\n"
              << "AVG copy time: " << avg_copy_duration << "\n"
              << "AVG execution time: " << avg_execution_duration << "\n"
              << "AVG result copy time: " << avg_result_copy_duration << "\n"
              << "Total time: " << total_timer.duration() << std::endl;

    out_file << avg_copy_duration << ","
             << avg_execution_duration << ","
             << avg_result_copy_duration << ","
             << total_timer.duration() << std::endl;

    std::cout << correct_inferences << std::endl;

    out_file.flush();
    out_file.close();

    std::ofstream execution_duration_file{"execution_times.csv", std::ios::out | std::ios::trunc};
    std::ostream_iterator<long> out_iterator{execution_duration_file, "\n"};
    std::copy(execution_duration_vector.begin(), execution_duration_vector.end(), out_iterator);
    execution_duration_file.flush();
    execution_duration_file.close();

    std::cout << "Benchmark done.\n";

}

void benchmark_CNN_single_split(dnnl::engine engine, dnnl::stream engine_stream,
                                const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> &dataset,
                                const std::vector<std::vector<float>> &cnn_weights,
                                const std::vector<std::vector<float>> &cnn_biases) {

    std::ofstream out_file{"unsafe_cnn_single_split_16.csv", std::ios::out | std::ios::trunc};

    out_file << "copy_time,exec_time,back_time,total_time\n";

    NeuralNetwork cnn = create_CNN(engine, engine_stream, cnn_weights, cnn_biases);
    std::cout << "CNN created.\n";

    auto image_count = dataset.training_images.size();
    std::vector<std::vector<float>> results{};
    results.reserve(image_count);
    int correct_inferences = 0;

    long total_copy_duration = 0;
    long total_execution_duration = 0;
    long total_result_copy_duration = 0;
    std::vector<long> execution_duration_vector{};

    Timers::StopTimer total_timer{};
    for (auto const &image: dataset.training_images) {
        Timers::StopTimer timer{};
        cnn.writeInput(image);
        timer.stop();
        total_copy_duration += timer.duration();
        timer.reset();
        cnn.execute();
        timer.stop();
        total_execution_duration += timer.duration();
        execution_duration_vector.emplace_back(timer.duration());
        timer.reset();
        results.emplace_back(cnn.readOutput());
        timer.stop();
        total_result_copy_duration += timer.duration();
        auto result_index = argmax(results[results.size() - 1]);
        auto correct_label = dataset.training_labels[results.size() - 1];

        correct_inferences += static_cast<int>(correct_label == result_index);
    }
    total_timer.stop();

    auto avg_copy_duration = total_copy_duration / image_count;
    auto avg_execution_duration = total_execution_duration / image_count;
    auto avg_result_copy_duration = total_result_copy_duration / image_count;

    std::cout << "Batched inference done!\n"
              << "AVG copy time: " << avg_copy_duration << "\n"
              << "AVG execution time: " << avg_execution_duration << "\n"
              << "AVG result copy time: " << avg_result_copy_duration << "\n"
              << "Total time: " << total_timer.duration() << std::endl;

    out_file << avg_copy_duration << ","
             << avg_execution_duration << ","
             << avg_result_copy_duration << ","
             << total_timer.duration() << std::endl;

    std::cout << correct_inferences << std::endl;

    out_file.flush();
    out_file.close();

    std::ofstream execution_duration_file{"cnn_execution_times.csv", std::ios::out | std::ios::trunc};
    std::ostream_iterator<long> out_iterator{execution_duration_file, "\n"};
    std::copy(execution_duration_vector.begin(), execution_duration_vector.end(), out_iterator);
    execution_duration_file.flush();
    execution_duration_file.close();

    std::cout << "Benchmark done.\n";

}

void benchmark_unsafe_threads_cnn(dnnl::engine engine, dnnl::stream engine_stream,
                                  const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> &dataset,
                                  const std::vector<std::vector<float>> &cnn_weights,
                                  const std::vector<std::vector<float>> &cnn_biases) {

    NeuralNetwork cnn = create_CNN(engine, engine_stream, cnn_weights, cnn_biases);
    std::cout << "CNN created.\n";

    std::ofstream out_file{"unsafe_threads_cnn_results.csv", std::ios::out | std::ios::trunc};

    out_file << "num_threads,time\n";
    out_file << "num_threads,time\n";

    for (int num_threads = 1; num_threads <= 64; ++num_threads) {
        omp_set_num_threads(num_threads);
        std::cout << num_threads << std::endl;
        int correct_inferences = 0;
        {
            Timers::BlockTimer bt{std::to_string(num_threads), ",", false, out_file};
            for (int i = 0; i < dataset.training_images.size(); ++i) {
                auto result_vector = cnn.infer(dataset.training_images[i]);
                auto result_index = argmax(result_vector);
                auto correct_label = dataset.training_labels[i];

                correct_inferences += static_cast<int>(correct_label == result_index);
            }
        }
        out_file.flush();
    }
    out_file.close();
    std::cout << "Benchmark done.\n";
}

void benchmark_pytorch_comparison(dnnl::engine engine, dnnl::stream engine_stream,
                                  std::ofstream &out_file,
                                  const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> &dataset,
                                  const std::vector<std::vector<float>> &mlp_weights,
                                  const std::vector<std::vector<float>> &mlp_biases,
                                  const std::vector<std::vector<float>> &cnn_weights,
                                  const std::vector<std::vector<float>> &cnn_biases) {

    std::cout << "Benchmarking without SGX" << std::endl;

    std::vector<int> thread_counts{1, 4, 16, 32};
    std::vector<long> batch_sizes{1, 16, 1024};

    std::vector<BatchedDataset> batched_datasets{};

    std::transform(batch_sizes.begin(),
                   batch_sizes.end(),
                   std::back_inserter(batched_datasets),
                   [&dataset](size_t batch_size) {
                       return BatchedDataset{dataset.training_images, dataset.training_labels, batch_size};
                   });

    std::vector<NeuralNetwork> mlps{};
    std::transform(batch_sizes.begin(), batch_sizes.end(), std::back_inserter(mlps),
                   [&engine, &engine_stream, &mlp_weights, &mlp_biases](long batch_size) {
                       return create_MLP(engine, engine_stream, mlp_weights, mlp_biases, batch_size);
                   });

    std::vector<NeuralNetwork> cnns{};
    std::transform(batch_sizes.begin(), batch_sizes.end(), std::back_inserter(cnns),
                   [&engine, &engine_stream, &cnn_weights, &cnn_biases](long batch_size) {
                       return create_CNN(engine, engine_stream, cnn_weights, cnn_biases, batch_size);
                   });

    std::vector<std::pair<std::string, std::vector<NeuralNetwork> &>> networks{{"mlp", mlps},
                                                                               {"cnn", cnns}};

    Timers::StopTimer timer{};

    for (auto &network_pair: networks) {
        for (auto thread_count: thread_counts) {
            omp_set_num_threads(thread_count);
            std::cout << "Number of threads: " << thread_count << std::endl;

            for (int batch_size_i = 0; batch_size_i < batch_sizes.size(); ++batch_size_i) {
                auto batch_size = batch_sizes[batch_size_i];
                auto &nn = network_pair.second[batch_size_i];
                auto &batched_dataset = batched_datasets[batch_size_i];
                std::cout << "Batch size: " << batch_size << std::endl;

                std::vector<long> inference_times{};
                inference_times.reserve(REPETITIONS);

                int correct_inferences = 0;
                for (int test_index = 0; test_index < REPETITIONS; ++test_index) {
                    timer.reset();
                    auto inference_result = nn.infer(batched_dataset.batched_data.at(test_index));
                    timer.stop();
                    inference_times.push_back(timer.duration());
                    correct_inferences += batched_dataset.check_batch(inference_result, test_index);
                }

                calculate_and_save_results(out_file, network_pair.first, thread_count, batch_size, inference_times,
                                           false);

                std::cout << "Correct inferences: " << correct_inferences << std::endl;
            }
        }
    }
}

// TODO: Remove the dependency on loading the weights
void native_benchmark(dnnl::engine &engine, dnnl::stream &engine_stream,
                      std::ofstream &out_file,
                      std::unique_ptr<const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t>> dataset,
                      std::unique_ptr<const std::vector<std::vector<float>>> mlp_weights,
                      std::unique_ptr<const std::vector<std::vector<float>>> mlp_biases,
                      std::unique_ptr<const std::vector<std::vector<float>>> cnn_weights,
                      std::unique_ptr<const std::vector<std::vector<float>>> cnn_biases,
                      const std::vector<std::string> &networks,
                      const std::vector<long> &batch_sizes,
                      const std::vector<int> &thread_counts) {

    std::cout << "Benchmarking without SGX" << std::endl;

    Timers::StopTimer timer{};

    for (const auto &network: networks) {
        std::cout << "Network: " << network << std::endl;

        for (int batch_size_i = 0; batch_size_i < batch_sizes.size(); ++batch_size_i) {
            auto batch_size = batch_sizes.at(batch_size_i);

            std::unique_ptr<NeuralNetwork> nn;
            if (network == "mlp" && mlp_weights && mlp_biases) {
                nn = std::make_unique<NeuralNetwork>(create_MLP(engine, engine_stream, *mlp_weights, *mlp_biases, batch_size));
            } else if (network == "cnn" && cnn_weights && cnn_biases) {
                nn = std::make_unique<NeuralNetwork>(create_CNN(engine, engine_stream, *cnn_weights, *cnn_biases, batch_size));
            } else {
                nn = std::make_unique<NeuralNetwork>(create_NN(engine, engine_stream, network, batch_size));
            }

            // TODO build dataset back in
            BatchedDataset batched_dataset;
            if (std::find(SMALL_INPUT_NETWORKS.begin(), SMALL_INPUT_NETWORKS.end(), network) != SMALL_INPUT_NETWORKS.end()) {
                batched_dataset = BatchedDataset{static_cast<size_t>(batch_size), 28 * 28, REPETITIONS};
            } else if (std::find(BIG_INPUT_NETWORKS.begin(), BIG_INPUT_NETWORKS.end(), network) != BIG_INPUT_NETWORKS.end()) {
                batched_dataset = BatchedDataset {static_cast<size_t>(batch_size), 3 * 224 * 224, REPETITIONS};
            } else {
                std::cout << "Creating batched dataset failed, stopping benchmark!" << std::endl;
                return;
            }

            std::cout << "Batch size: " << batch_size << std::endl;

            for (auto thread_count: thread_counts) {
                omp_set_num_threads(thread_count);
                std::cout << "Number of threads: " << thread_count << std::endl;

                std::vector<long> inference_times{};
                inference_times.reserve(REPETITIONS);

                int correct_inferences = 0;
                for (int test_index = 0; test_index < REPETITIONS; ++test_index) {
                    timer.reset();
                    auto inference_result = nn->infer(batched_dataset.batched_data.at(test_index));
                    timer.stop();
                    inference_times.push_back(timer.duration());
                    correct_inferences += batched_dataset.check_batch(inference_result, test_index);
                }

                calculate_and_save_results_copy(out_file, network, thread_count, batch_size, inference_times,
                                                false, false);

                //std::cout << "Correct inferences: " << correct_inferences << std::endl;
            }
        }
    }
}

void native_benchmark(dnnl::engine &engine, dnnl::stream &engine_stream,
                                      std::ofstream &out_file,
                                      const std::vector<std::string> &networks,
                                      const std::vector<long> &batch_sizes,
                                      const std::vector<int> &thread_counts) {
    native_benchmark(engine, engine_stream, out_file, nullptr, nullptr, nullptr, nullptr, nullptr, networks,
                     batch_sizes, thread_counts);
}

void benchmark_alexnet(dnnl::engine &engine, dnnl::stream &engine_stream) {
    NeuralNetwork alexnet = create_simple_AlexNet(engine, engine_stream, 1);

    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};  // Generates random integers
    std::normal_distribution<float> normal_distribution;

    auto nd_pos_gen = [&normal_distribution, &mersenne_engine]() {
        return std::abs(normal_distribution(mersenne_engine));
    };

    initMemoryWithRandomValues(alexnet.getInputMemory(), nd_pos_gen);

    Timers::StopTimer timer{};

    for (int i = 0; i < REPETITIONS; ++i) {
        initMemoryWithRandomValues(alexnet.getInputMemory(), nd_pos_gen);
        timer.reset();
        alexnet.execute();
        timer.stop();
        std::cout << std::to_string(timer.duration()) << std::endl;
    }

}

void benchmark_vgg19(dnnl::engine &engine, dnnl::stream &engine_stream) {
    NeuralNetwork vgg19 = create_VGG19(engine, engine_stream, 1);

    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};  // Generates random integers
    std::normal_distribution<float> normal_distribution;

    auto nd_pos_gen = [&normal_distribution, &mersenne_engine]() {
        return std::abs(normal_distribution(mersenne_engine));
    };

    initMemoryWithRandomValues(vgg19.getInputMemory(), nd_pos_gen);

    Timers::StopTimer timer{};

    for (int i = 0; i < REPETITIONS; ++i) {
        initMemoryWithRandomValues(vgg19.getInputMemory(), nd_pos_gen);
        timer.reset();
        vgg19.execute();
        timer.stop();
        std::cout << std::to_string(timer.duration()) << std::endl;
    }

}

#endif //SGX_DNNL_UNSAFEBENCHMARKS_HPP
