#!/bin/bash

# ----------------------------------------------------------------------------------------------------------------------
# Generate ensemble of configuration files for Xanthos from MIT climate data scenario, model, realizations
#
# TO RUN:
# sbatch /rcfs/projects/gcims/projects/mit_climate/code/gcims_exp1_workflow/scripts/generate_config_files.py.sl
# ----------------------------------------------------------------------------------------------------------------------

#SBATCH -n 1
#SBATCH -t 00:05:00
#SBATCH -A IHESD
#SBATCH -J xancfg
#SBATCH -p short

# Load Modules
module load gcc/11.2.0
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh

# activate conda environment
conda activate xanthosenv

start=`date +%s.%N`

python /rcfs/projects/gcims/projects/mit_climate/code/gcims_exp1_workflow/scripts/generate_config_files.py

end=`date +%s.$N`

runtime=$( echo "($end - $start) / 60" | bc -l )

echo "Run completed in $runtime minutes"