#!/bin/bash
#SBATCH --account=acf-utk0011
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess.out
#SBATCH --error=preprocess.err
#SBATCH --time=00:01:00
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
set -euo pipefail

echo "Starting job on $(hostname)"
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"

apptainer exec container.sif python3 preprocess.py /nfs/home/gflanag4/M-Competition_DSE-512/dataset/archive 3/Weekly-train.csv 

echo "Timestamp: $(date)"