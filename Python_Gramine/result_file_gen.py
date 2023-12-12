with open("results", "a") as results:
    results.write("sgx" + ',' +
                  "nn" + "," + "threads" +
                  "," + "batch_size" +
                  "," + "inference_time" + "\n")
results.close()