# gcims_exp1_workflow
GCIMS Experiment 1 Workflow

## Processing environment

### Cluster
Processing of the climate data and Xanthos simulations were conducted on a PNNL High Performance Computing (HPC) cluster named Deception (deception.pnl.gov) with an initial configuration consists of 96 compute nodes boasting a total of 6,144 CPU cores powered by dual AMD EPYC 7502 CPUs. The CPUs run at 2.5 GHz with the ability to boost to 3.35 GHz. 

### Shell
The SLURM scripts are written to execute from a BASH shell.

### Python 
We used a miniconda distribution of Python version 3.9.5.  This can be loaded on Deception via:
```bash
module load gcc/11.2.0
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
```

We also created a virtual environment which that also includes the Xanthos install and other needed packages.  This can be activated by:
```bash
conda activate xanthosenv
```

## Pre-processing

### Xanthos calibration

#### Source data
Describe and cite source data chosen and give the location of where this is on Deception.

#### Calibration process
Xanthos was calibrated to GSWP3 using the following steps.
- fill in

Xanthos configuration outputs were written here:  `/rcfs/projects/gcims/models/xanthos/calibration_gswp3_watergap2/data/outputs/gswp3_20220717`

#### Calibration post-processing
The parameter files for each individual basin were then consolidated into the input file that Xanthos expects by running:
```bash
sbatch consolidate_calibrated_params.sl
```
This command triggers the `scripts/consolidate_calibrated_params.py` Python file.

### Quality control of climate data
Describe steps for the QC of the climate data.

### Process historical data
Convert the historical baseline data into the input format that Xanthos expects.  This data will be stitched to the future data during the future simulations.
Historical data can be created by running:
```bash
sbatch generate_historical_climate.sl
```
This command triggers the `scripts/generate_historical_climate.py` Python file.

## Simulation workflow

### Generate configuration files
Generate configuration files for each scenario, model, realization combination.  This uses the `data/template.ini` file found in this repository.
Configuration files are generated by executing the following on Deception from here `/rcfs/projects/gcims/projects/mit_climate/code/gcims_exp1_workflow/scripts`:
```bash
sbatch generate_config_files.sl
```
This command triggers the `scripts/generate_config_files.py` Python file.

### Process future data and run Xanthos per realization
This step does the following for each scenario, model, realization:
- converts the climate data to the format that Xanthos expects and stitches the historical period to the start 
- runs xanthos and generates outputs for:
  - potential evapotranspiration (PET) in units km<sup>3</sup>/month
  - actual evapotranspiration (AET) in units km<sup>3</sup>/month
  - runoff (Q) in units km<sup>3</sup>/month
  - soil moisture in units km<sup>3</sup>/month
  - average channel flow in units m<sup>3</sup>/second
  - basin aggregated runoff in units km<sup>3</sup>/month
  - country aggregated runoff in units km<sup>3</sup>/month
  - GCAM region aggregated runoff in units km<sup>3</sup>/month
  - drought severity, intensity, and duration
- deletes the converted climate data 

### Convert the Xanthos outputs to the XML structure needed for GCAM
Convert the Xanthos output files to the XML structure needed for GCAM

### Run GCAM
Run GCAM

## Post-processing quality control
Describe steps for the QC of the Xanthos and GCAM output data.
