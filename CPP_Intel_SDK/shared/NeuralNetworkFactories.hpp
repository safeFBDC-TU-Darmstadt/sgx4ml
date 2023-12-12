//
// Created by adrian on 02.12.22.
//

#ifndef SGX_DNNL_NEURALNETWORKFACTORIES_HPP
#define SGX_DNNL_NEURALNETWORKFACTORIES_HPP

#include <array>
#include "dnnl_utils.hpp"
#include "NeuralNetwork.hpp"

constexpr const std::array<const char*, 4> SMALL_INPUT_NETWORKS {"mlp", "medium_mlp", "large_mlp", "cnn"};
constexpr const std::array<const char*, 4> BIG_INPUT_NETWORKS {"simple_alex", "alex", "vgg16", "vgg19"};


NeuralNetwork create_MLP(dnnl::engine &engine, dnnl::stream &engine_stream,
                         long batch_size = 1) {
    NeuralNetwork mlp{engine, engine_stream, dnnl::memory::format_tag::nc,
                      dnnl::memory::data_type::f32, dnnl::memory::dims{batch_size, 28 * 28}};

    mlp.addFullyConnectedLayer(100, PostOp::Sigmoid);
    mlp.addFullyConnectedLayer(10, PostOp::None);
    return mlp;
}

NeuralNetwork create_MLP(dnnl::engine &engine, dnnl::stream &engine_stream,
                         const std::vector<std::vector<float>> &mlp_weights,
                         const std::vector<std::vector<float>> &mlp_biases,
                         long batch_size = 1) {
    if (mlp_weights.size() != 2 || mlp_biases.size() != 2) {
        throw std::runtime_error(
                "Error in NeuralNetworkFactories::create_MLP: Unexpected length of mlp_weights or mlp_biases.");
    }
    NeuralNetwork mlp{engine, engine_stream, dnnl::memory::format_tag::nc,
                      dnnl::memory::data_type::f32, dnnl::memory::dims{batch_size, 28 * 28}};

    mlp.addFullyConnectedLayer(100, PostOp::Sigmoid, &mlp_weights[0], &mlp_biases[0]);
    mlp.addFullyConnectedLayer(10, PostOp::None, &mlp_weights[1], &mlp_biases[1]);
    return mlp;
}

NeuralNetwork create_large_MLP1(dnnl::engine &engine, dnnl::stream &engine_stream, long batch_size = 1) {
    NeuralNetwork mlp{engine, engine_stream, dnnl::memory::format_tag::nc,
                      dnnl::memory::data_type::f32, dnnl::memory::dims{batch_size, 28 * 28}};

    mlp.addFullyConnectedLayer(2000, PostOp::ReLU);
    mlp.addFullyConnectedLayer(1000, PostOp::ReLU);
    mlp.addFullyConnectedLayer(10, PostOp::Sigmoid);

    return mlp;
}

NeuralNetwork create_large_MLP2(dnnl::engine &engine, dnnl::stream &engine_stream, long batch_size = 1) {
    NeuralNetwork mlp{engine, engine_stream, dnnl::memory::format_tag::nc,
                      dnnl::memory::data_type::f32, dnnl::memory::dims{batch_size, 28 * 28}};

    mlp.addFullyConnectedLayer(4000, PostOp::ReLU);
    mlp.addFullyConnectedLayer(2000, PostOp::ReLU);
    mlp.addFullyConnectedLayer(1000, PostOp::ReLU);
    mlp.addFullyConnectedLayer(10, PostOp::Sigmoid);

    return mlp;
}

NeuralNetwork create_CNN(dnnl::engine &engine, dnnl::stream &engine_stream,
                         long batch_size = 1) {
    NeuralNetwork cnn{engine, engine_stream, dnnl::memory::format_tag::nchw, dnnl::memory::data_type::f32,
                      {batch_size, 1, 28, 28}};
    cnn.addConvolutionalLayer(6, {5, 5}, {1, 1}, {0, 0}, PostOp::ReLU);
    cnn.addPoolingLayer({2, 2}, {2, 2}, {0, 0});
    cnn.addConvolutionalLayer(12, {5, 5}, {1, 1}, {0, 0}, PostOp::ReLU);
    cnn.addPoolingLayer({2, 2}, {2, 2}, {0, 0});
    cnn.addFullyConnectedLayer(120, PostOp::ReLU, nullptr, nullptr,
                               dnnl::memory::format_tag::oihw);
    cnn.addFullyConnectedLayer(60, PostOp::ReLU);
    cnn.addFullyConnectedLayer(10, PostOp::ReLU);
    return cnn;
}

NeuralNetwork create_CNN(dnnl::engine &engine, dnnl::stream &engine_stream,
                         const std::vector<std::vector<float>> &cnn_weights,
                         const std::vector<std::vector<float>> &cnn_biases,
                         long batch_size = 1) {
    if (cnn_weights.size() != 5 || cnn_biases.size() != 5) {
        throw std::runtime_error(
                "Error in NeuralNetworkFactories::create_MLP: Unexpected length of cnn_weights or cnn_biases.");
    }
    NeuralNetwork cnn{engine, engine_stream, dnnl::memory::format_tag::nchw, dnnl::memory::data_type::f32,
                      {batch_size, 1, 28, 28}};
    cnn.addConvolutionalLayer(6, {5, 5}, {1, 1}, {0, 0}, PostOp::ReLU, &cnn_weights[0], &cnn_biases[0]);
    cnn.addPoolingLayer({2, 2}, {2, 2}, {0, 0});
    cnn.addConvolutionalLayer(12, {5, 5}, {1, 1}, {0, 0}, PostOp::ReLU, &cnn_weights[1], &cnn_biases[1]);
    cnn.addPoolingLayer({2, 2}, {2, 2}, {0, 0});
    cnn.addFullyConnectedLayer(120, PostOp::ReLU, &cnn_weights[2], &cnn_biases[2],
                               dnnl::memory::format_tag::oihw);
    cnn.addFullyConnectedLayer(60, PostOp::ReLU, &cnn_weights[3], &cnn_biases[3]);
    cnn.addFullyConnectedLayer(10, PostOp::ReLU, &cnn_weights[4], &cnn_biases[4]);
    return cnn;
}

NeuralNetwork create_simple_AlexNet(dnnl::engine &engine, dnnl::stream &engine_stream, long batch_size = 1) {
    NeuralNetwork alexnet{engine, engine_stream, dnnl::memory::format_tag::nchw, dnnl::memory::data_type::f32,
                          {batch_size, 3, 224, 224}};

    alexnet.addConvolutionalLayer(96, {11, 11}, {4, 4}, {2, 2}, PostOp::ReLU);
    alexnet.addPoolingLayer({3, 3}, {2, 2});
    alexnet.addConvolutionalLayer(256, {5, 5}, {1, 1}, {2, 2}, PostOp::ReLU);
    alexnet.addPoolingLayer({3, 3}, {2, 2});
    alexnet.addConvolutionalLayer(384, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    alexnet.addConvolutionalLayer(384, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    alexnet.addConvolutionalLayer(256, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    alexnet.addPoolingLayer({3, 3}, {2, 2});
    alexnet.addFullyConnectedLayer(500, PostOp::ReLU, nullptr, nullptr, dnnl::memory::format_tag::oihw);

    return alexnet;
}

NeuralNetwork create_AlexNet(dnnl::engine &engine, dnnl::stream &engine_stream, long batch_size = 1) {
    NeuralNetwork alexnet{engine, engine_stream, dnnl::memory::format_tag::nchw, dnnl::memory::data_type::f32,
                          {batch_size, 3, 224, 224}};

    alexnet.addConvolutionalLayer(96, {11, 11}, {4, 4}, {2, 2}, PostOp::ReLU);
    alexnet.addPoolingLayer({3, 3}, {2, 2});
    alexnet.addConvolutionalLayer(256, {5, 5}, {1, 1}, {2, 2}, PostOp::ReLU);
    alexnet.addPoolingLayer({3, 3}, {2, 2});
    alexnet.addConvolutionalLayer(384, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    alexnet.addConvolutionalLayer(384, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    alexnet.addConvolutionalLayer(256, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    alexnet.addPoolingLayer({3, 3}, {2, 2});
    alexnet.addFullyConnectedLayer(4096, PostOp::ReLU, nullptr, nullptr,
                                   dnnl::memory::format_tag::oihw);
    alexnet.addFullyConnectedLayer(4096, PostOp::ReLU);
    alexnet.addFullyConnectedLayer(1000, PostOp::ReLU);

    return alexnet;
}

NeuralNetwork create_VGG16(dnnl::engine &engine, dnnl::stream &engine_stream, long batch_size = 1) {
    NeuralNetwork vgg16{engine, engine_stream, dnnl::memory::format_tag::nchw, dnnl::memory::data_type::f32,
                        {batch_size, 3, 224, 224}};

    vgg16.addConvolutionalLayer(64, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addConvolutionalLayer(64, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addPoolingLayer({2, 2}, {2, 2}, {0, 0});

    vgg16.addConvolutionalLayer(128, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addConvolutionalLayer(128, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addPoolingLayer({2, 2}, {2, 2}, {0, 0});

    vgg16.addConvolutionalLayer(256, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addConvolutionalLayer(256, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addConvolutionalLayer(256, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addPoolingLayer({2, 2}, {2, 2}, {0, 0});

    vgg16.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addPoolingLayer({2, 2}, {2, 2}, {0, 0});

    vgg16.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg16.addPoolingLayer({2, 2}, {2, 2}, {0, 0});

    vgg16.addFullyConnectedLayer(4096, PostOp::ReLU, nullptr, nullptr,
                                 dnnl::memory::format_tag::oihw);
    vgg16.addFullyConnectedLayer(4096, PostOp::ReLU);
    vgg16.addFullyConnectedLayer(1000, PostOp::ReLU);

    return vgg16;
}

NeuralNetwork create_VGG19(dnnl::engine &engine, dnnl::stream &engine_stream, long batch_size = 1) {
    NeuralNetwork vgg19{engine, engine_stream, dnnl::memory::format_tag::nchw, dnnl::memory::data_type::f32,
                        {batch_size, 3, 224, 224}};

    vgg19.addConvolutionalLayer(64, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addConvolutionalLayer(64, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addPoolingLayer({2, 2}, {2, 2}, {0, 0});

    vgg19.addConvolutionalLayer(128, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addConvolutionalLayer(128, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addPoolingLayer({2, 2}, {2, 2}, {0, 0});

    vgg19.addConvolutionalLayer(256, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addConvolutionalLayer(256, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addConvolutionalLayer(256, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addConvolutionalLayer(256, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addPoolingLayer({2, 2}, {2, 2}, {0, 0});

    vgg19.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addPoolingLayer({2, 2}, {2, 2}, {0, 0});

    vgg19.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addConvolutionalLayer(512, {3, 3}, {1, 1}, {1, 1}, PostOp::ReLU);
    vgg19.addPoolingLayer({2, 2}, {2, 2}, {0, 0});

    vgg19.addFullyConnectedLayer(4096, PostOp::ReLU, nullptr, nullptr,
                                 dnnl::memory::format_tag::oihw);
    vgg19.addFullyConnectedLayer(4096, PostOp::ReLU);
    vgg19.addFullyConnectedLayer(1000, PostOp::ReLU);

    return vgg19;
}

NeuralNetwork create_NN(dnnl::engine &engine, dnnl::stream &engine_stream, const std::string& name, long batch_size) {
    if (name == "mlp") {
        return create_MLP(engine, engine_stream, batch_size);
    } else if (name == "medium_mlp") {
        return create_large_MLP1(engine, engine_stream, batch_size);
    } else if (name == "large_mlp") {
        return create_large_MLP2(engine, engine_stream, batch_size);
    } else if (name == "cnn") {
        return create_CNN(engine, engine_stream, batch_size);
    } else if (name == "simple_alex") {
        return create_simple_AlexNet(engine, engine_stream, batch_size);
    } else if (name == "alex") {
        return create_AlexNet(engine, engine_stream, batch_size);
    } else if (name == "vgg16") {
        return create_VGG16(engine, engine_stream, batch_size);
    } else if (name == "vgg19") {
        return create_VGG19(engine, engine_stream, batch_size);
    } else {
        throw std::runtime_error("Error in NeuralNetworkFactories::create_NN: Unknown neural network name.");
    }
}

NeuralNetwork create_NN(dnnl::engine &engine, dnnl::stream &engine_stream, const std::string& name, long batch_size,
                        const std::vector<std::vector<float>> &weights,
                        const std::vector<std::vector<float>> &biases) {
    if (name == "mlp") {
        return create_MLP(engine, engine_stream, weights, biases, batch_size);
    } else if (name == "cnn") {
        return create_CNN(engine, engine_stream, weights, biases, batch_size);
    } else {
        throw std::runtime_error("Error in NeuralNetworkFactories::create_NN: Something with the trained networks went wrong.");
    }
}

#endif //SGX_DNNL_NEURALNETWORKFACTORIES_HPP
