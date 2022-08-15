#!/bin/bash

# ----------------------------------------------------------------------------------------------------------------------
# Simulate future climate runs through Xanthos.  Processes 72 simulations for each scenario, model and 50 realizations
#  per node in parallel.
#
# TO RUN:
# sbatch --array=0-72 /rcfs/projects/gcims/projects/mit_climate/code/gcims_exp1_workflow/scripts/simulate_future_climate.sl
# ----------------------------------------------------------------------------------------------------------------------

#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A IHESD
#SBATCH -J xansim
#SBATCH -p short

# Load Modules
module load gcc/11.2.0
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh

# activate conda environment
conda activate xanthosenv

start=`date +%s.%N`

python /rcfs/projects/gcims/projects/mit_climate/code/gcims_exp1_workflow/scripts/simulate_future_climate.py $SLURM_ARRAY_TASK_ID

end=`date +%s.$N`

runtime=$( echo "($end - $start) / 60" | bc -l )

echo "Run completed in $runtime minutes"
