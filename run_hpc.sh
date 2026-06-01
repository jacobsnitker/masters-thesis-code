#!/bin/bash
#SBATCH --job-name=clare_replication
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu
#SBATCH --time=3-00:00:00
#SBATCH --partition=acltr
#SBATCH --mem=128G

# Usage: sbatch run_hpc.sh [combo_start] [combo_end] [script] [extra_args]
# e.g.:  sbatch run_hpc.sh 0 5 main.py
#        sbatch run_hpc.sh 0 5 main_2s.py
#        sbatch run_hpc.sh 0 15 measure_inference_time.py --dl
COMBO_START=${1:-0}
COMBO_END=${2:-15}
SCRIPT=${3:-main_2s.py}
EXTRA_ARGS=${4:-}

echo "Running on $(hostname):"
nvidia-smi

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/itu/easybuild/software/CUDA/12.1.1

# ── Load modules ──────────────────────────────────────────────────────────────
module --ignore_cache load Python/3.11.3-GCCcore-12.3.0
module --ignore_cache load cuDNN/8.9.2.26-CUDA-12.1.1

# ── Build venv if it doesn't exist ────────────────────────────────────────────
if [ ! -f /home/jacv/venv/bin/python3 ]; then
    echo "Building venv..."
    python3 -m venv /home/jacv/venv
    source /home/jacv/venv/bin/activate
    pip install --upgrade pip
    pip install tensorflow==2.15.0 numpy">=1.24.0,<2.0.0" pandas scipy scikit-learn lightgbm xgboost neurokit2 antropy tqdm
else
    echo "Venv exists, activating..."
    source /home/jacv/venv/bin/activate
fi

echo "Pip list:"
pip list | grep -E "tensorflow|keras|tqdm|numpy|scipy|sklearn|lightgbm|xgboost|neurokit|antropy"

# ── Run ───────────────────────────────────────────────────────────────────────
cd /home/jacv/clare
mkdir -p logs

echo "Job started: $(date)"
echo "Python: $(python3 --version)"
echo "TensorFlow GPU available: $(python3 -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))')"
echo "Combos: ${COMBO_START} to ${COMBO_END}"

if [ "${SCRIPT}" = "measure_inference_time.py" ]; then
    python3 ${SCRIPT} ${EXTRA_ARGS}
else
    python3 ${SCRIPT} --scheme both --combo-start ${COMBO_START} --combo-end ${COMBO_END} ${EXTRA_ARGS}
fi

echo "Job finished: $(date)"
