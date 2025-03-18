#!/bin/bash

NUM_IMAGES=50000
NUM_BATCHES=125
BATCH_SIZE=16
NUM_JOBS=$(( NUM_IMAGES / NUM_BATCHES / BATCH_SIZE ))
SEED=42
SAMPLING_TEMPERATURE_ANNEALING=False

export_str="ALL,SEED=${SEED},NUM_BATCHES=${NUM_BATCHES},BATCH_SIZE=${BATCH_SIZE},SAMPLING_TEMPERATURE_ANNEALING=${SAMPLING_TEMPERATURE_ANNEALING}"
for NUM_ITER in 32; do
  for DECODING_STRATEGY in "remdm_rescale" "remdm_cap"; do
    for MASK_SCHEDULING_METHOD in "uniform"; do
      for SAMPLING_TEMPERATURE in 0.6; do
        for REMDM_ETA in 0.01 0.02 0.05; do
          echo "***********************************************************************************************************"
          echo "Scheduling jobs for: DECODING_STRATEGY=${DECODING_STRATEGY},REMDM_ETA=${REMDM_ETA},NUM_ITER=${NUM_ITER},MASK_SCHEDULING_METHOD=${MASK_SCHEDULING_METHOD},SAMPLING_TEMPERATURE=${SAMPLING_TEMPERATURE}"
          export_str="${export_str},DECODING_STRATEGY=${DECODING_STRATEGY},REMDM_ETA=${REMDM_ETA},NUM_ITER=${NUM_ITER},MASK_SCHEDULING_METHOD=${MASK_SCHEDULING_METHOD},SAMPLING_TEMPERATURE=${SAMPLING_TEMPERATURE}"
          for i in $(seq 0 $((NUM_JOBS - 1))); do
            echo Job ID $((i + 1)): sbatch \
              --export="${export_str},OUTPUT_START_INDEX=$((i * NUM_BATCHES * BATCH_SIZE))" \
              --job-name="${DECODING_STRATEGY}_eta-${REMDM_ETA}_${MASK_SCHEDULING_METHOD}_n-${NUM_ITER}_samp-temp-anneal-${SAMPLING_TEMPERATURE_ANNEALING}_samp-temp-${SAMPLING_TEMPERATURE}_$((i * NUM_BATCHES * BATCH_SIZE))-$(((i + 1) * NUM_BATCHES * BATCH_SIZE - 1))" \
              run_gen_imgs.sh
            sbatch \
              --export="${export_str},OUTPUT_START_INDEX=$((i * NUM_BATCHES * BATCH_SIZE))" \
              --job-name="${DECODING_STRATEGY}_eta-${REMDM_ETA}_${MASK_SCHEDULING_METHOD}_n-${NUM_ITER}_samp-temp-anneal-${SAMPLING_TEMPERATURE_ANNEALING}_samp-temp-${SAMPLING_TEMPERATURE}_$((i * NUM_BATCHES * BATCH_SIZE))-$(((i + 1) * NUM_BATCHES * BATCH_SIZE - 1))" \
              run_gen_imgs.sh
          done
          echo "***********************************************************************************************************"
          echo ""
        done
      done
    done
  done
done
