#!/bin/bash -l
#SBATCH --job-name=loss_curves_cv
#SBATCH --output=results/loss_curves_cv_%j.log
#SBATCH --error=results/loss_curves_cv_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

export PYTHONPATH=~/clare/venv/lib/python3.11/site-packages:$PYTHONPATH
cd ~/clare
~/clare/venv/bin/python3.11 plot_loss_curves_cv.py
