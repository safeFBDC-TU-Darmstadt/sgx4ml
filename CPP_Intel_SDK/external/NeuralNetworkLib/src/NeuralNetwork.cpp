//
// Created by adrian on 26.07.22.
//

#include "NeuralNetwork.hpp"
#include <algorithm>
#include <random>
#include "dnnl_utils.hpp"

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

NeuralNetwork::NeuralNetwork(dnnl::engine &engine,
                             dnnl::stream &engine_stream,
                             tag input_memory_tag,
                             dnnl::memory::data_type data_type,
                             const dnnl::memory::dims &input_dims) :
        engine{engine},
        engine_stream{engine_stream},
        dt{data_type},
        input_dims{input_dims},
        input_size{std::accumulate(input_dims.begin(), input_dims.end(), 1, std::multiplies<>())},
        last_dims{input_dims},
        last_md{input_dims, dt, input_memory_tag},
        last_memory{last_md, engine},
        user_input_memory{last_memory},
        batch_size{input_dims[0]} {}

Layer &NeuralNetwork::getLayer(int position) {
    return layers[position];
}

void NeuralNetwork::addLayer(const Layer &layer) {
    layers.push_back(layer);
}

size_t NeuralNetwork::layerCount() const {
    return layers.size();
}

dnnl::memory &NeuralNetwork::getInputMemory() {
    return user_input_memory;
}


int NeuralNetwork::getOutputSize() const {
    return std::accumulate(last_md.data.dims, last_md.data.dims + last_md.data.ndims, 1, std::multiplies<>());
}

long NeuralNetwork::getBatchSize() const {
    return batch_size;
}

void NeuralNetwork::addFullyConnectedLayer(int layer_size, PostOp post_op,
                                           const std::vector<float> *const weights,
                                           const std::vector<float> *const biases,
                                           dnnl::memory::format_tag weights_memory_tag) {
    dnnl::memory::dims src_dims = last_dims;
    dnnl::memory::dims weights_dims = {layer_size};
    // Add all dimensions from src_dims except the batch size to weights
    weights_dims.insert(weights_dims.end(), ++src_dims.begin(), src_dims.end());
    dnnl::memory::dims bias_dims = {layer_size};
    dnnl::memory::dims dst_dims = {batch_size, layer_size};

    auto src_md = dnnl::memory::desc(src_dims, dt, tag::any);
    auto weights_md = dnnl::memory::desc(weights_dims, dt, tag::any);
    auto bias_md = dnnl::memory::desc(bias_dims, dt, tag::any);
    auto dst_md = dnnl::memory::desc(dst_dims, dt, tag::any);

    auto inner_product_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_training, src_md,
                                                                weights_md, bias_md, dst_md);

    /// Create postop

    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;

    dnnl::primitive_attr inner_product_attr;
    if (post_op != PostOp::None) {
        dnnl::post_ops inner_product_ops;
        // Default postop is relu, option is sigmoid
        auto algorithm = dnnl::algorithm::eltwise_relu;
        if (post_op == PostOp::Sigmoid) {
            algorithm = dnnl::algorithm::eltwise_logistic;
        }
        inner_product_ops.append_eltwise(scale, algorithm, alpha, beta);
        inner_product_attr.set_post_ops(inner_product_ops);
    }
    auto inner_product_pd = dnnl::inner_product_forward::primitive_desc(inner_product_desc, inner_product_attr, engine);

    auto src_memory = last_memory;
    if (inner_product_pd.src_desc() != last_memory.get_desc()) {
        src_memory = dnnl::memory(inner_product_pd.src_desc(), engine);
        layers.push_back({dnnl::reorder(last_memory, src_memory),
                          {{DNNL_ARG_FROM, last_memory}, {DNNL_ARG_TO, src_memory}}});
    }

    auto weights_memory = dnnl::memory(inner_product_pd.weights_desc(), engine);
    auto bias_memory = dnnl::memory(inner_product_pd.bias_desc(), engine);
    auto dst_memory = dnnl::memory(inner_product_pd.dst_desc(), engine);

    // TODO make the setup of the random number generator somehow static for the NN
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};  // Generates random integers
    std::normal_distribution<float> normal_distribution {0.01};

    auto nd_gen = [&normal_distribution, &mersenne_engine]() {
        return normal_distribution(mersenne_engine);
    };

    auto user_weights_md = dnnl::memory::desc(weights_dims, dt, weights_memory_tag);
    auto user_weights_memory = dnnl::memory(user_weights_md, engine);

    if (weights != nullptr) {
        write_to_dnnl_memory(weights->data(), user_weights_memory);
    } else {
        initMemoryWithRandomValues(user_weights_memory, nd_gen);
    }

    dnnl::reorder(user_weights_memory, weights_memory)
            .execute(engine_stream, {
                    {DNNL_ARG_FROM, user_weights_memory},
                    {DNNL_ARG_TO,   weights_memory}
            });

    auto user_bias_md = dnnl::memory::desc(bias_dims, dt, tag::x);
    auto user_bias_memory = dnnl::memory(user_bias_md, engine);

    if (biases != nullptr) {
        write_to_dnnl_memory(biases->data(), user_bias_memory);
    } else {
        initMemoryWithRandomValues(user_bias_memory, nd_gen);
    }

    dnnl::reorder(user_bias_memory, bias_memory)
            .execute(engine_stream, {
                    {DNNL_ARG_FROM, user_bias_memory},
                    {DNNL_ARG_TO,   bias_memory}
            });

    auto inner_product_prim = dnnl::inner_product_forward(inner_product_pd);

    std::unordered_map<int, dnnl::memory> inner_product_args;
    inner_product_args.emplace(DNNL_ARG_SRC, src_memory);
    inner_product_args.emplace(DNNL_ARG_WEIGHTS, weights_memory);
    inner_product_args.emplace(DNNL_ARG_BIAS, bias_memory);
    inner_product_args.emplace(DNNL_ARG_DST, dst_memory);

    addLayer({inner_product_prim, inner_product_args});

    last_dims = dst_dims;
    last_md = inner_product_pd.dst_desc();
    last_memory = dst_memory;
}

void NeuralNetwork::addConvolutionalLayer(int output_channels, dnnl::memory::dims kernel_dims,
                                          dnnl::memory::dims strides, dnnl::memory::dims padding, PostOp post_op,
                                          const std::vector<float> *const weights,
                                          const std::vector<float> *const biases) {
    // Assumes that last_dims.size() >= 4 and order is NCHW
    auto input_channels = last_dims[1];
    auto input_height = last_dims[2];
    auto input_width = last_dims[3];

    // Calculation from DNNL documentation, simplified, because we do not support asymmetric padding
    // https://oneapi-src.github.io/oneDNN/dev_guide_convolution.html#doxid-dev-guide-convolution
    auto result_height = static_cast<int>(
            (static_cast<double>(input_height + 2 * padding[0] - kernel_dims[0]) /
             static_cast<double>(strides[0])) + 1);
    auto result_width = static_cast<int>(
            (static_cast<double>(input_width + 2 * padding[1] - kernel_dims[1]) /
             static_cast<double>(strides[1])) + 1);

    dnnl::memory::dims src_dims = {batch_size, input_channels, input_height, input_width};
    dnnl::memory::dims weights_dims = {output_channels, input_channels, kernel_dims[0], kernel_dims[1]};
    dnnl::memory::dims bias_dims = {output_channels};
    dnnl::memory::dims dst_dims = {batch_size, output_channels, result_height, result_width};

    auto src_md = dnnl::memory::desc(src_dims, dt, tag::any);
    auto weights_md = dnnl::memory::desc(weights_dims, dt, tag::any);
    auto bias_md = dnnl::memory::desc(bias_dims, dt, tag::any);
    auto dst_md = dnnl::memory::desc(dst_dims, dt, tag::any);

    auto conv_desc = dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
                                                     dnnl::algorithm::convolution_direct, src_md, weights_md,
                                                     bias_md, dst_md, strides, padding,
                                                     padding);

    /// Create postop

    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;

    dnnl::primitive_attr conv_attr;
    if (post_op != PostOp::None) {
        dnnl::post_ops conv_ops;
        // Default postop is relu, option is sigmoid
        auto algorithm = dnnl::algorithm::eltwise_relu;
        if (post_op == PostOp::Sigmoid) {
            algorithm = dnnl::algorithm::eltwise_logistic;
        }
        conv_ops.append_eltwise(scale, algorithm, alpha, beta);
        conv_attr.set_post_ops(conv_ops);
    }

    auto conv_pd = dnnl::convolution_forward::primitive_desc(conv_desc, conv_attr, engine);

    auto src_memory = last_memory;
    if (conv_pd.src_desc() != last_memory.get_desc()) {
        src_memory = dnnl::memory(conv_pd.src_desc(), engine);
        layers.push_back({dnnl::reorder(last_memory, src_memory),
                          {{DNNL_ARG_FROM, last_memory}, {DNNL_ARG_TO, src_memory}}});
    }

    auto weights_memory = dnnl::memory(conv_pd.weights_desc(), engine);
    auto bias_memory = dnnl::memory(conv_pd.bias_desc(), engine);
    auto dst_memory = dnnl::memory(conv_pd.dst_desc(), engine);

    // TODO make the setup of the random number generator somehow static for the NN
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};  // Generates random integers
    std::normal_distribution<float> normal_distribution {0.01};

    auto nd_gen = [&normal_distribution, &mersenne_engine]() {
        return normal_distribution(mersenne_engine);
    };

    auto user_weights_md = dnnl::memory::desc(weights_dims, dt, tag::oihw);
    auto user_weights_memory = dnnl::memory(user_weights_md, engine);

    if (weights != nullptr) {
        write_to_dnnl_memory(weights->data(), user_weights_memory);
    } else {
        initMemoryWithRandomValues(user_weights_memory, nd_gen);
    }

    dnnl::reorder(user_weights_memory, weights_memory)
            .execute(engine_stream, {
                    {DNNL_ARG_FROM, user_weights_memory},
                    {DNNL_ARG_TO,   weights_memory}
            });

    auto user_bias_md = dnnl::memory::desc(bias_dims, dt, tag::x);
    auto user_bias_memory = dnnl::memory(user_bias_md, engine);

    if (biases != nullptr) {
        write_to_dnnl_memory(biases->data(), user_bias_memory);
    } else {
        initMemoryWithRandomValues(user_bias_memory, nd_gen);
    }

    dnnl::reorder(user_bias_memory, bias_memory)
            .execute(engine_stream, {
                    {DNNL_ARG_FROM, user_bias_memory},
                    {DNNL_ARG_TO,   bias_memory}
            });

    addLayer({dnnl::convolution_forward(conv_pd),
              {
                      {DNNL_ARG_SRC, src_memory},
                      {DNNL_ARG_WEIGHTS, weights_memory},
                      {DNNL_ARG_BIAS, bias_memory},
                      {DNNL_ARG_DST, dst_memory}
              }
             });

    last_memory = dst_memory;
    last_md = conv_pd.dst_desc();
    last_dims = dst_dims;
}

void
NeuralNetwork::addPoolingLayer(dnnl::memory::dims kernel_dims, dnnl::memory::dims strides, dnnl::memory::dims padding) {
    // Assumes that last_dims.size() >= 4 and order is NCHW
    auto input_channels = last_dims[1];
    auto input_height = last_dims[2];
    auto input_width = last_dims[3];

    // Calculation from DNNL documentation, simplified, because we do not support asymmetric padding
    // https://oneapi-src.github.io/oneDNN/dev_guide_convolution.html#doxid-dev-guide-convolution
    auto result_height = static_cast<int>(
            (static_cast<double>(input_height + 2 * padding[0] - kernel_dims[0]) /
             static_cast<double>(strides[0])) + 1);
    auto result_width = static_cast<int>(
            (static_cast<double>(input_width + 2 * padding[1] - kernel_dims[1]) /
             static_cast<double>(strides[1])) + 1);

    dnnl::memory::dims src_dims = {batch_size, input_channels, input_height, input_width};
    dnnl::memory::dims dst_dims = {batch_size, input_channels, result_height, result_width};

    auto src_md = dnnl::memory::desc(src_dims, dt, tag::nchw);
    auto dst_md = dnnl::memory::desc(dst_dims, dt, tag::any);

    auto pooling_desc = dnnl::pooling_forward::desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_max,
                                                    src_md, dst_md, strides, kernel_dims, padding, padding);

    auto pooling_pd = dnnl::pooling_forward::primitive_desc(pooling_desc, engine);

    auto src_memory = last_memory;
    if (pooling_pd.src_desc() != last_memory.get_desc()) {
        src_memory = dnnl::memory(pooling_pd.src_desc(), engine);
        layers.push_back({dnnl::reorder(last_memory, src_memory),
                          {{DNNL_ARG_FROM, last_memory}, {DNNL_ARG_TO, src_memory}}});
    }
    auto dst_memory = dnnl::memory(pooling_pd.dst_desc(), engine);

    addLayer({dnnl::pooling_forward(pooling_pd),
              {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_DST, dst_memory}}});

    last_memory = dst_memory;
    last_md = pooling_pd.dst_desc();
    last_dims = dst_dims;
}

void NeuralNetwork::writeInput(const std::vector<float> &input_data) {
    if (input_data.size() != input_size) {
        throw std::runtime_error(
                "Error in NeuralNetwork::writeInput: Input vector size does not match expected input size.");
    }

    reset_dnnl_memory((void *) input_data.data(), getInputMemory());
}

void NeuralNetwork::writeInput(const float *begin, size_t input_count) {
    if (input_count != input_size) {
        throw std::runtime_error(
                "Error in NeuralNetwork::writeInput: Input value count does not match expected input size.");
    }

    reset_dnnl_memory((void *) begin, getInputMemory());
}

void NeuralNetwork::execute() const {
    for (const auto & layer : layers) {
        layer.prim.execute(engine_stream, layer.args);
    }
}

std::vector<float> NeuralNetwork::readOutput() const {
    auto output_size = getOutputSize();
    std::vector<float> output(output_size);
    auto output_memory = layers.at(layers.size() - 1).args.at(DNNL_ARG_DST);
    auto output_memory_handle = static_cast<float *>(output_memory.get_data_handle());

    std::copy(output_memory_handle, output_memory_handle + output_size, output.begin());
    return output;
}

void NeuralNetwork::readOutput(float *result_pointer, size_t result_count) {
    if (result_count != getOutputSize()) {
        throw std::runtime_error(
                "Error in NeuralNetwork::readOutput: Output size does not equal buffer size.");
    }

    auto output_memory = layers.at(layers.size() - 1).args.at(DNNL_ARG_DST);
    auto output_memory_handle = static_cast<float *>(output_memory.get_data_handle());

    std::copy(output_memory_handle, output_memory_handle + result_count, result_pointer);
}

std::vector<float> NeuralNetwork::infer(const std::vector<float> &input) {
    writeInput(input);
    execute();
    return readOutput();
}

std::vector<float> NeuralNetwork::infer(const float *begin, size_t input_count) {
    writeInput(begin, input_count);
    execute();
    return readOutput();
}

void NeuralNetwork::infer(const float *begin, size_t input_count, float *result_pointer, size_t result_count) {
    writeInput(begin, input_count);
    execute();
    readOutput(result_pointer, result_count);
}


