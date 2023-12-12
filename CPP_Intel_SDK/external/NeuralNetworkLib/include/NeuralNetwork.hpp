//
// Created by adrian on 26.07.22.
//

#ifndef DNNLFASHIONMLP_NEURALNETWORK_HPP
#define DNNLFASHIONMLP_NEURALNETWORK_HPP

#include <dnnl.hpp>

struct Layer {
    dnnl::primitive prim;
    std::unordered_map<int, dnnl::memory> args;
};

enum class PostOp {
    None = 0,
    ReLU = 1,
    Sigmoid = 2
};

class NeuralNetwork {
private:
    dnnl::engine &engine;
    dnnl::stream &engine_stream;
    dnnl::memory::data_type dt;
    dnnl::memory::dims input_dims;
    dnnl::memory::dim input_size;
    dnnl::memory::dims last_dims;
    dnnl::memory::desc last_md;
    dnnl::memory last_memory;
    dnnl::memory user_input_memory;
    long batch_size;
    std::vector<Layer> layers;
public:
    NeuralNetwork(dnnl::engine &engine, dnnl::stream &engine_stream,
                  dnnl::memory::format_tag input_memory_tag,
                  dnnl::memory::data_type data_type, const dnnl::memory::dims &input_dims);

    Layer &getLayer(int);

    void addLayer(const Layer &layer);

    size_t layerCount() const;

    dnnl::memory &getInputMemory();

    int getOutputSize() const;

    long getBatchSize() const;

    void addFullyConnectedLayer(int layer_size, PostOp post_op = PostOp::None,
                                const std::vector<float> *const weights = nullptr,
                                const std::vector<float> *const biases = nullptr,
                                dnnl::memory::format_tag weights_memory_tag = dnnl::memory::format_tag::oi);

    void addConvolutionalLayer(int output_channels, dnnl::memory::dims kernel_dims, dnnl::memory::dims strides = {1, 1},
                               dnnl::memory::dims padding = {0, 0}, PostOp post_op = PostOp::None,
                               const std::vector<float> *const weights = nullptr,
                               const std::vector<float> *const biases = nullptr);

    void addPoolingLayer(dnnl::memory::dims kernel_dims, dnnl::memory::dims strides = {1, 1},
                         dnnl::memory::dims padding = {0, 0});

    void addReLU();

    void writeInput(const std::vector<float> &);

    void writeInput(const float *begin, size_t input_count);

    void execute() const;

    std::vector<float> readOutput() const;

    void readOutput(float *result_pointer, size_t result_count);

    /**
     * Combines writeInput, execute and readOutput in one function.
     * @param input The inpout for the first layer of the NN. Is copied to the input memory.
     * @return A vector containing the output of the last layer of the NN.
     */
    std::vector<float> infer(const std::vector<float> &input);

    std::vector<float> infer(const float *begin, size_t input_count);

    void infer(const float *begin, size_t input_count, float *result_pointer, size_t result_count);
};


#endif //DNNLFASHIONMLP_NEURALNETWORK_HPP
