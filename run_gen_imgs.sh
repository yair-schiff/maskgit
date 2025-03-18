#!/bin/bash
#SBATCH -o ./watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                         # Total number of nodes requested
#SBATCH --get-user-env               # retrieve the users login environment
#SBATCH --mem=32000                  # server memory requested (per node)
#SBATCH -t 96:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu
#SBATCH --constraint="[a100|a6000|a5000|3090]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

# Expecting:
#  - SEED
#  - NUM_BATCHES
#  - BATCH_SIZE
#  - DECODING_STRATEGY
#  - REMDM_ETA
#  - NUM_ITER
#  - MASK_SCHEDULING_METHOD
#  - OUTPUT_START_INDEX
#  - SAMPLING_TEMPERATURE_ANNEALING
#  - SAMPLING_TEMPERATURE

# Setup environment
source "${CONDA_SHELL}"
if [ -z "${CONDA_PREFIX}" ]; then
  conda activate maskgit_env
 elif [[ "${CONDA_PREFIX}" != *"/maskgit_env" ]]; then
  conda deactivate
  conda activate maskgit_env
fi

OUTPUT_DIR="${PWD}/outputs/${DECODING_STRATEGY}/num_iter-${NUM_ITER}_mask_sched-${MASK_SCHEDULING_METHOD}_samp-temp-anneal-${SAMPLING_TEMPERATURE_ANNEALING}_samp-temp-${SAMPLING_TEMPERATURE}_eta-${REMDM_ETA}"
mkdir -p "${OUTPUT_DIR}"

python gen_imgs.py \
  --output_image_path "${OUTPUT_DIR}/images" \
  --output_label_path "${OUTPUT_DIR}/labels" \
  --output_start_index ${OUTPUT_START_INDEX} \
  --seed ${SEED} \
  --num_batches ${NUM_BATCHES} \
  --batch_size ${BATCH_SIZE} \
  --image_size 256 \
  --decoding_strategy "${DECODING_STRATEGY}" \
  --remdm_eta ${REMDM_ETA} \
  --num_iter ${NUM_ITER} \
  --mask_scheduling_method "${MASK_SCHEDULING_METHOD}" \
  --sampling_temperature_annealing ${SAMPLING_TEMPERATURE_ANNEALING} \
  --sampling_temperature ${SAMPLING_TEMPERATURE}
