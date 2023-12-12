import sys
import time
import torch

models = {
    'mlp': "output/saved_mlp_model.pt",
    'cnn': "output/saved_cnn_model.pt",
    'alex': "output/saved_alexnet_model.pt",
    'alex_simple' : "output/saved_alexnet_simple.pt",
    'vgg16': "output/saved_vgg16_model.pt",
    'vgg19': "output/saved_vgg19_model.pt",
    # 'cnn_new': "output/saved_cnn_new_model.pt",  # Not implemented in C++
    'large_mlp1': "output/saved_large_mlp1_model.pt",  # aka medium_mlp
    'large_mlp2': "output/saved_large_mlp2_model.pt",  # aka large_mlp
    # simple_alex : missing!
}


def gen_data(model, batch_size):
    if model == 'cnn' or model == "cnn_new":
        return torch.rand((batch_size, 1, 28, 28,))
    elif model == 'mlp' or model == 'large_mlp1' or model == 'large_mlp2':
        return torch.rand((batch_size, 28 * 28,))
    elif model == 'alex' or model == 'alex_simple':
        return torch.rand((batch_size, 3, 227, 227,))
    else:
        return torch.rand((batch_size, 3, 224, 224,))


def run_inferences(threads, batch_sizes, iterations, sgx_flag):
    # print("hi")
    sgx_str = {True: "yes", False: "no"}
    with open("results", "a") as results:
        for m in models:
            # print(models[m])
            model = torch.load(models[m])
            model.eval()
            for batch_size in batch_sizes:
                for i in range(iterations):
                    input_data = gen_data(m, batch_size)
                    start_time = time.perf_counter_ns()
                    output = model(input_data)
                    end_time = time.perf_counter_ns()
                    exec_time = end_time - start_time
                    if i % 10 == 0:
                        print(f"Time to infer model {m} with {threads:2d} threads "
                              f"and {batch_size:04d} sized batch: {exec_time:10.7f}")

                    results.write(sgx_str[sgx_flag] + ',' +
                                  m + "," + str(threads) +
                                  "," + str(batch_size) +
                                  "," + str(exec_time) + "\n")


if __name__ == "__main__":
    sgx_flag = bool(int(sys.argv[1]))
    threads = int(sys.argv[2])
    iterations = int(sys.argv[3])
    batch_sizes = [16, 1]
    # print(sgx_flag)
    if not sgx_flag:
        torch.set_num_threads(threads)
    run_inferences(threads, batch_sizes, iterations, sgx_flag)
