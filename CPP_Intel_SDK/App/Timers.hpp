//
// Created by adrian on 19.08.22.
//

#ifndef SGX_DNNL_TIMERS_HPP
#define SGX_DNNL_TIMERS_HPP

#include <chrono>
#include <string>

namespace Timers {

    class BlockTimer {
        std::string what;
        std::string separator{" Time: "};
        bool print_unit{true};
        std::chrono::high_resolution_clock::time_point start{std::chrono::high_resolution_clock::now()};
    public:
        BlockTimer() = delete;

        BlockTimer(const BlockTimer &other) = delete;

        BlockTimer(BlockTimer &&other) = delete;

        BlockTimer &operator=(const BlockTimer &other) = delete;

        BlockTimer &operator=(BlockTimer &&other) = delete;

        BlockTimer(std::string what, std::ostream &output) : what{std::move(what)}, output{output.rdbuf()} {}

        BlockTimer(std::string what, std::string separator, bool print_unit, std::ostream &output) : what{
                std::move(what)}, separator{std::move(separator)}, output{output.rdbuf()},
                print_unit{print_unit} {}

        ~BlockTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            output << what << separator << duration << (print_unit ? " ns\n" : "\n");
        }
    private:
        std::ostream output;
    };

    class StopTimer {
        std::chrono::high_resolution_clock::time_point start_time{std::chrono::high_resolution_clock::now()};
        std::chrono::high_resolution_clock::time_point end_time{start_time};
    public:
        StopTimer() = default;

        void reset() {
            start_time = std::chrono::high_resolution_clock::now();
            end_time = start_time;
        }

        void start() {
            reset();
        }

        void stop() {
            end_time = std::chrono::high_resolution_clock::now();
        }

        auto duration() const {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        }
    };

}

#endif //SGX_DNNL_TIMERS_HPP
