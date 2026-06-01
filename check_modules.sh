#!/bin/bash
#SBATCH --job-name=check_modules
#SBATCH --output=job.check.%j.out
#SBATCH --error=job.check.%j.err
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --time=00:05:00
#SBATCH --partition=acltr

echo "=== Available Python modules ==="
module avail Python 2>&1

echo "=== System Python ==="
which python3
python3 --version
pip3 --version

echo "=== Conda ==="
which conda 2>/dev/null || echo "no conda"
which mamba 2>/dev/null || echo "no mamba"
