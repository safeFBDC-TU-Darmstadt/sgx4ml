//
// Created by adrian on 11.11.22.
//

#ifndef SGX_DNNL_EXPERIMENTHELPERS_HPP
#define SGX_DNNL_EXPERIMENTHELPERS_HPP

void
calculate_and_save_results(std::ofstream &out_file, const std::string& network, int thread_count, unsigned long batch_size,
                           std::vector<long> &inference_times, bool sgx) {
    auto max = *std::max_element(inference_times.begin(), inference_times.end());
    auto min = *std::min_element(inference_times.begin(), inference_times.end());
    auto avg = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) /
               static_cast<double>(REPETITIONS);

    // Write raw inference times to file
    std::string inference_times_file_name {"inference_times_"
                                           + std::string(sgx ? "sgx" : "nosgx") + "_"
                                           + network + "_"
                                           + std::to_string(thread_count) + "_"
                                           + std::to_string(batch_size)
                                           + ".csv"};
    std::ofstream inference_time_ofs{inference_times_file_name, std::ios::out | std::ios::trunc};
    std::ostream_iterator<long> inference_time_writer(inference_time_ofs, "\n");
    std::copy(inference_times.begin(), inference_times.end(), inference_time_writer);

    std::nth_element(inference_times.begin(), inference_times.begin() + (REPETITIONS / 2),
                     inference_times.end());
    auto med = inference_times.at(REPETITIONS / 2);

    out_file << (sgx ? "yes," : "no,")
             << network << ","
             << thread_count << ","
             << batch_size << ","
             << min << ","
             << avg << ","
             << med << ","
             << max << std::endl;
}

void
calculate_and_save_results_combined(std::ofstream &out_file, const std::string& network, int thread_count,
                                    unsigned long batch_size, std::vector<long> &inference_times, bool sgx) {
    for (auto inference_time : inference_times) {
        out_file << (sgx ? "yes," : "no,")
                 << network << ","
                 << thread_count << ","
                 << batch_size << ","
                 << inference_time << "\n";
    }
}

void
calculate_and_save_results_copy(std::ofstream &out_file, const std::string& network, int thread_count,
                                    unsigned long batch_size, std::vector<long> &inference_times, bool sgx, bool copy) {
    for (auto inference_time : inference_times) {
        out_file << (sgx ? "yes," : "no,")
                 << (copy ? "yes," : "no,")
                 << network << ","
                 << thread_count << ","
                 << batch_size << ","
                 << inference_time << "\n";
    }
}

#endif //SGX_DNNL_EXPERIMENTHELPERS_HPP
