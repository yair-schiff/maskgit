#!/bin/bash

NUM_IMAGES=50000
NUM_BATCHES=125
BATCH_SIZE=16
NUM_JOBS=$(( NUM_IMAGES / NUM_BATCHES / BATCH_SIZE ))
SEED=42

export_str="ALL,SEED=${SEED},NUM_BATCHES=${NUM_BATCHES},BATCH_SIZE=${BATCH_SIZE}"
for NUM_ITER in 256; do
  for DECODING_STRATEGY in "mdim" "maskgit" "mdlm"; do
    for MASK_SCHEDULING_METHOD in "cosine"; do
      echo "***********************************************************************************************************"
      echo "Scheduling jobs for: DECODING_STRATEGY=${DECODING_STRATEGY},NUM_ITER=${NUM_ITER},MASK_SCHEDULING_METHOD=${MASK_SCHEDULING_METHOD}"
      export_str="${export_str},DECODING_STRATEGY=${DECODING_STRATEGY},NUM_ITER=${NUM_ITER},MASK_SCHEDULING_METHOD=${MASK_SCHEDULING_METHOD}"
      for i in $(seq 0 $((NUM_JOBS - 1))); do
        echo Job ID $((i + 1)): sbatch \
          --export="${export_str},OUTPUT_START_INDEX=$((i * NUM_BATCHES * BATCH_SIZE))" \
          --job-name="${DECODING_STRATEGY}_${MASK_SCHEDULING_METHOD}_n-${NUM_ITER}_$((i * NUM_BATCHES * BATCH_SIZE))-$(((i + 1) * NUM_BATCHES * BATCH_SIZE - 1))" \
          run_gen_imgs.sh
        sbatch \
          --export="${export_str},OUTPUT_START_INDEX=$((i * NUM_BATCHES * BATCH_SIZE))" \
          --job-name="${DECODING_STRATEGY}_${MASK_SCHEDULING_METHOD}_n-${NUM_ITER}_$((i * NUM_BATCHES * BATCH_SIZE))-$(((i + 1) * NUM_BATCHES * BATCH_SIZE - 1))" \
          run_gen_imgs.sh
      done
      echo "***********************************************************************************************************"
      echo ""
    done
  done
done