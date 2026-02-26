#!/bin/bash
#SBATCH --account=acf-utk0011
#SBATCH --job-name=m4_train
#SBATCH --array=0-10
#SBATCH --output=logs/train_%a.out
#SBATCH --error=logs/train_%a.err
#SBATCH --time=00:10:00
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
set -euo pipefail

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"
echo "Processing series: W${SLURM_ARRAY_TASK_ID}"
echo "numX's: 13"

# Adding this section for Part D of the assignment to simulate a failure for series W7
if [ "$SLURM_ARRAY_TASK_ID" -eq 7 ]; then
    echo "INTENTIONAL FAILURE: Simulated missing input for series W7."
    echo "This failure is deliberate for Part D of the assignment."
    exit 1
fi

mkdir -p logs
apptainer exec container.sif python3 create_model.py $SLURM_ARRAY_TASK_ID 13

echo "Timestamp: $(date)"