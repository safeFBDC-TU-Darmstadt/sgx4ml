'''
1, 4, 16, 32 Threads not pinned
batch size 1, 16, 1024
min(inference time) avg(inference time) max(inference time) over the first 100 images in the training set.
MLP, CNN
Inside SGX, Outside SGX

4 * 3 * 2 * 2 = 48 setups
'''
import sys
import time
import csv

import torch
from old_files.FashionMNIST_data import get_training_data, get_training_dataloader

model_files = {
    'mlp': "output/saved_mlp_model.pt",
    'cnn': "output/saved_cnn_model.pt"
}
models = ["mlp", "cnn"]

outfile = open("times/" + "result.csv", "w")

writer = csv.writer(outfile)
head_row = ['Model', 'Batch Size', 'Min Time', 'Max Time', 'Avg Time']
writer.writerow(head_row)

train_data = get_training_data()

for m in models:
    model = torch.load(model_files[m])
    model.eval()
    for batch_size in [1, 16, 1024]:
        iterations = 100
        max_time = 0
        min_time = sys.maxsize
        sum_time = 0
        train_file = open("times/" + "training_" + m + "_" + str(batch_size) + ".csv", "w")

        train_writer = csv.writer(train_file)
        train_writer.writerow(["Time"])
        for itr, (features, targets) in enumerate(get_training_dataloader(batch_size, False)):
            if itr >= iterations:
                break
            # img, labels = train_data[itr * batch_size:itr * batch_size + batch_size]
            if "mlp" == m:
                input_ = features.view(-1, 28 * 28)
            else:
                input_ = features
            start_time = time.time()
            output = model(input_)
            end_time = time.time()
            exec_time = end_time - start_time
            if itr % 10 == 0:
                print("Time to infer with {:04d} sized batch:{:10.7f}".format(batch_size,
                                                                              exec_time))
            train_writer.writerow([f'{exec_time:.7f}'])
            if itr != 0:
                max_time = max(max_time, exec_time)
                min_time = min(min_time, exec_time)
                sum_time = sum_time + exec_time
        train_file.close()
        avg_time = sum_time / min(iterations - 1, itr)
        print("======================================")
        print("{:4d} batch size: min time {:10.7f}, max time {:10.7f}, avg time {:10.7f}"
              .format(batch_size, min_time, max_time, avg_time))
        writer.writerow([m, batch_size, f'{min_time:.7f}', f'{max_time:.7f}', f'{avg_time:.7f}'])
        print("======================================")
