#!/bin/sh
rm results
iterations=50
python3 result_file_gen.py
make clean
make SGX=1
for thread in 1 2 4 8 16
do
  # sed -i "s/loader.env.OMP_NUM_THREADS =.*/loader.env.OMP_NUM_THREADS = \"$thread\"/" pytorch.manifest.template
  export OMP_NUM_THREADS="$thread"
  echo "$OMP_NUM_THREADS"
  gramine-sgx ./pytorch -B ./Inference_exp_new.py 1 $thread $iterations
  python3 -B Inference_exp_new.py 0 $thread $iterations
done
