# gcims_exp1_workflow
GCIMS Experiment 1 Workflow

## Pre-processing quality control
Describe steps for the QC of the climate data.

## Simulation workflow

### Generate configuration files
Generate configuration files for each scenario, model, realization combination.  This uses the `data/template.ini` file found in this repository.

### Process historical data
Convert the historical baseline data into the input format that Xanthos expects.  This data will be stitched to the future data during the future simulations.

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
