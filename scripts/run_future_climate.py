import os
import calendar
import glob
import itertools
import sys

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from xanthos import Xanthos


def get_xanthos_coordinates(xanthos_reference_file: str) -> np.ndarray:
    """Generate an array of xanthos latitude, longitude values from the input xanthos reference file.

    :param xanthos_reference_file:                  Full path with file name and extension to the xanthos reference file.
    :type xanthos_reference_file:                   str

    :returns:                                       Array of latitude, longitude values corresponding with ordered
                                                    xanthos grid cells (67,420)

    """

    # read in the xanthos reference file to a data frame
    df = pd.read_csv(xanthos_reference_file)

    # generate an array of lat, lon for xanthos land grids
    return df[["latitude", "longitude"]].values


def generate_coordinate_reference(xanthos_lat_lon: np.ndarray,
                                  climate_lat_arr: np.ndarray,
                                  climate_lon_arr: np.ndarray):
    """Create a data frame of extracted data from the source climate product to the Xanthos
    input structure.

    :param xanthos_lat_lon:                         Array of latitude, longitude values corresponding with ordered
                                                    xanthos grid cells (67,420).
    :type xanthos_lat_lon:                          np.ndarray

    :param climate_lat_arr:                         Climate latitude array associated with each latitude from Xanthos.
    :type climate_lat_arr:                          np.ndarray

    :param climate_lon_arr:                         Climate longitude array associated with each longitude from Xanthos.
    :type climate_lon_arr:                          np.ndarray

    :returns:                                       [0] list of climate latitude index values associated with each
                                                        xanthos grid cell
                                                    [1] list of climate longitude index values associated with each
                                                        xanthos grid cell

    """

    climate_lat_idx = []
    climate_lon_idx = []

    # get the climate grid index associated with each xanthos grid centroid via lat, lon
    for index, coords in enumerate(xanthos_lat_lon):
        # break out lat, lon from xanthos coordinate pairs for each grid
        lat, lon = coords

        # get the index pair in the climate data associated with xanthos coordinates
        lat_idx = np.where(climate_lat_arr == lat)[0][0]
        lon_idx = np.where(climate_lon_arr == lon)[0][0]

        # append the climate grid index associated with each lat, lon from Xanthos
        climate_lat_idx.append(lat_idx)
        climate_lon_idx.append(lon_idx)

    return climate_lat_idx, climate_lon_idx


def get_days_in_month(start_year: int,
                      through_year: int) -> list:
    """Generate a list of the number of days in each month of the record of interest including leap years.

    :param start_year:                              Four digit start year
    :type start_year:                               int

    :param through_year:                            Four digit through year
    :type through_year:                             int

    :returns:                                       A list of days in the month for the period of interest.

    """

    days_in_month_standard = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_in_month_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    try:
        start_year = int(start_year)
    except ValueError:
        raise (
            f"Expected year in YYYY format as first position in underscore separated file name.  Received:  '{split_name[0]}'")

    try:
        through_year = int(through_year)
    except ValueError:
        raise (
            f"Expected year in YYYY format as second position in underscore separated file name.  Received:  '{split_name[1]}'")

    year_list = range(start_year, through_year + 1, 1)

    days_in_month = []
    for i in year_list:

        if calendar.isleap(i):
            days_in_month.extend(days_in_month_leap)
        else:
            days_in_month.extend(days_in_month_standard)

    return days_in_month


def extract_climate_data(ds: xr.Dataset,
                         target_variable_dict: dict,
                         climate_lat_idx: list,
                         climate_lon_idx: list) -> dict:
    """Extract target variables for each xanthos grid cell.

    :param ds:                                      Input xarray dataset from the climate NetCDF file.
    :type ds:                                       xr.Dataset

    :param target_variable_dict:                    Dictionary of variables to extract data for and their target units.
    :type target_variable_dict:                     dict

    :param climate_lat_idx:                         List of index values from the climate data corresponding with
                                                    xanthos grid cell latitudes.
    :type climate_lat_idx:                          list

    :param climate_lon_idx:                         List of index values from the climate data corresponding with
                                                    xanthos grid cell longitudes.
    :type climate_lon_idx:                          list

    :return:                                        A dictionary of variable to extracted data.

    """

    return {i: ds[i].values[:, climate_lat_idx, climate_lon_idx].T for i in target_variable_dict.keys()}


def run_extraction(climate_file: str,
                   xanthos_reference_file: str,
                   target_variables: dict,
                   pet_output_dir: str,
                   climate_output_dir: str,
                   scenario: str,
                   model: str,
                   start_year: int,
                   through_year: int,
                   stitch_to_historic: bool) -> list:
    """Workhorse function to extract target variables at each xanthos grid cell and write to a compressed
    numpy array.

    :param climate_file:                            Full path with file name and extension to the input climate file.
    :type climate_file:                             str

    :param xanthos_reference_file:                  Full path with file name and extension to the xanthos reference file.
    :type xanthos_reference_file:                   str

    :param target_variables:                        Dict of variables to extract data for and their target units.
    :type target_variables:                         dict

    :param pet_output_dir:                          Full path to the directory where the PET output files will be stored.
    :type pet_output_dir:                           str

    :param climate_output_dir:                      Full path to the directory where the climate runoff output files will be stored.
    :type climate_output_dir:                       str

    :param scenario:                                Scenario name to process.
    :type scenario:                                 str

    :param model:                                   Model name to process.
    :type model:                                    str

    :param start_year:                              Four digit start year
    :type start_year:                               int

    :param through_year:                            Four digit through year
    :type through_year:                             int

    :param stitch_to_historic:                      Choice to stitch historic data to the output
    :type stitch_to_historic:                       bool

    :returns:                                       List of full path with file name and extension to the output files

    """
    historic_file_structure = "{}__historic__baseclim__baseclim_0.5_1931_2010_allvar.npy"

    output_file_list = []

    # read in climate NetCDF to an xarray dataset
    ds = xr.open_dataset(climate_file)

    # generate an array of lat, lon for xanthos land grid cells
    xanthos_lat_lon = get_xanthos_coordinates(xanthos_reference_file)

    # generate lists of lat, lon indices from the climate data associated with xanthos grid cells
    climate_lat_idx, climate_lon_idx = generate_coordinate_reference(xanthos_lat_lon=xanthos_lat_lon,
                                                                     climate_lat_arr=ds.LAT.values,
                                                                     climate_lon_arr=ds.LON.values)

    # generate a dictionary of variable to extracted array of xanthos grid cell locations
    data = extract_climate_data(ds=ds,
                                target_variable_dict=target_variables,
                                climate_lat_idx=climate_lat_idx,
                                climate_lon_idx=climate_lon_idx)

    # convert units for temperature variables from K to C
    data["Tair"] += -273.15
    data["Tmin"] += -273.15

    # create output file name from input file
    basename = os.path.splitext(os.path.basename(climate_file))[0]

    # convert units for precipitation from mm/day to mm/month; assumes start month of January
    days_in_month_list = get_days_in_month(start_year, through_year)
    data["PRECTmmd"] *= days_in_month_list

    # write relevant files to the inputs/pet/penman_monteith directory for use by the PET model
    pet_variables = ["Hurs", "FLDS", "FSDS", "WIND", "Tair", "Tmin"]

    for varname in pet_variables:

        out_file = os.path.join(pet_output_dir, f"{varname}__{scenario}__{model}__{basename}.npy")

        # stitch to historic if so desired
        if stitch_to_historic:

            hist_arr = np.load(os.path.join(pet_output_dir, historic_file_structure.format(varname)))

            out_arr = np.concatenate([hist_arr, data[varname]], axis=1)

            np.save(out_file, out_arr)

        else:

            # write each as a NPY file in the PET directory
            np.save(out_file, data[varname])

        output_file_list.append(out_file)

    # write relevant files to the inputs/climate directory for use by the abcdm model
    runoff_variables = ["PRECTmmd", "Tmin"]

    for varname in runoff_variables:

        out_file = os.path.join(climate_output_dir, f"{varname}__{scenario}__{model}__{basename}.npy")

        # stitch to historic if so desired
        if stitch_to_historic:

            hist_arr = np.load(os.path.join(climate_output_dir, historic_file_structure.format(varname)))

            out_arr = np.concatenate([hist_arr, data[varname]], axis=1)

            np.save(out_file, out_arr)

        else:

            # write each as a NPY file in the PET directory
            np.save(out_file, data[varname])

        output_file_list.append(out_file)

    return output_file_list


def run_extraction_parallel(data_directory: str,
                            xanthos_reference_file: str,
                            target_variables: dict,
                            output_directory: str,
                            scenario: str,
                            model: str,
                            njobs=-1):
    """Extract target variables at each xanthos grid cell and write to a compressed
    numpy array for each file in parallel.

    :param data_directory:                          Directory containing the input climate data directory structure.
    :type data_directory:                           str

    :param xanthos_reference_file:                  Full path with file name and extension to the xanthos reference file.
    :type xanthos_reference_file:                   str

    :param target_variables:                        Dictionary of variables to extract data for and their target units.
    :type target_variables:                         dict

    :param output_directory:                        Full path to the directory where the output file will be stored.
    :type output_directory:                         str

    :param scenario:                                Scenario name to process.
    :type scenario:                                 str

    :param model:                                   Model name to process.
    :type model:                                    str
    """

    # get a list of target files to process in parallel
    target_files = glob.glob(os.path.join(data_directory, scenario, model, "*_0.5_e00*_monthly.nc"))

    print(f"Processing files:  {target_files}")

    # process all files for a model and scenario in parallel
    results = Parallel(n_jobs=njobs, backend="loky")(delayed(run_extraction)(climate_file=i,
                                                                             xanthos_reference_file=xanthos_reference_file,
                                                                             target_variables=target_variables,
                                                                             output_directory=output_directory,
                                                                             scenario=scenario,
                                                                             model=model) for i in target_files)

    return results


class ConfigGenerator:
    """Generate a Xanthos configuration file from inputs.


    :param template_config_file:                    Full path to the template configuration file with file name and
                                                    extension.
    :type template_config_file:                     str

    :param project_name:                            Name of the run.
    :type project_name:                             str

    :param root_directory:                          Root directory containing the xanthos input and output directories.
    :type root_directory:                           str

    :param start_year:                              Start year of xanthos simulation.
    :type start_year:                               int

    :param end_year:                                End year of xanthos simulation.
    :type end_year:                                 int

    :param pet_tas_filename:                        File name without path of the near surface air temperature climate
                                                    data.
    :type pet_tas_filename:                         str

    :param pet_tmin_filename:                       File name without path of the monthly mean of daily minimum near
                                                    surface air temperature climate data.
    :type pet_tmin_filename:                        str

    :param pet_rhs_filename:                        File name without path of the near surface relative humidity climate
                                                    data.
    :type pet_rhs_filename:                         str

    :param pet_rlds_filename:                       File name without path of the surface incident longwave radiation
                                                    climate data.
    :type pet_rlds_filename:                        str

    :param pet_rsds_filename:                       File name without path of the surface incident shortwave radiation
                                                    climate data.
    :type pet_rsds_filename:                        str

    :param pet_wind_filename:                       File name without path of the near surface wind speed climate data.
    :type pet_wind_filename:                        str

    :param runoff_params_filename:                  File name without path of the xanthos parameter file from
                                                    calibration.
    :type runoff_params_filename:                   str

    :param runoff_tmin_file:                        Full path with file name and extension to the monthly mean of daily
                                                    minimum near surface air temperature climate data.
    :type runoff_tmin_file:                         str

    :param runoff_pr_file:                          Full path with file name and extension to the monthly precipitation
                                                    rate climate data.
    :type runoff_pr_file:                           str

    :param output_file:                             Full path with file name and extension to the output ini file.
    :type output_file:                              str

    """

    PROJECT_NAME_KEY = "<PROJECT_NAME>"
    ROOT_DIRECTORY_KEY = "<ROOT_DIR>"
    START_YEAR_KEY = "<START_YEAR>"
    END_YEAR_KEY = "<END_YEAR>"

    PET_TAS_KEY = "<PET_TAS_KEY>"
    PET_TMIN_KEY = "<PET_TMIN_KEY>"
    PET_RHS_KEY = "<PET_RHS_KEY>"
    PET_RLDS_KEY = "<PET_RLDS_KEY>"
    PET_RSDS_KEY = "<PET_RSDS_KEY>"
    PET_WIND_KEY = "<PET_WIND_KEY>"

    RUNOFF_PARAMS_KEY = "<RUNOFF_PARAMS_KEY>"
    RUNOFF_TMIN_KEY = "<RUNOFF_TMIN_KEY>"
    RUNOFF_PR_KEY = "<RUNOFF_PR_KEY>"

    def __init__(self,
                 template_config_file: str,
                 project_name: str,
                 root_directory: str,
                 start_year: int,
                 end_year: int,
                 pet_tas_filename: str,
                 pet_tmin_filename: str,
                 pet_rhs_filename: str,
                 pet_rlds_filename: str,
                 pet_rsds_filename: str,
                 pet_wind_filename: str,
                 runoff_params_filename: str,
                 runoff_tmin_file: str,
                 runoff_pr_file: str,
                 output_file: str):
        self.template_config_file = template_config_file
        self.project_name = project_name
        self.root_directory = root_directory
        self.start_year = str(start_year)
        self.end_year = str(end_year)
        self.pet_tas_filename = pet_tas_filename
        self.pet_tmin_filename = pet_tmin_filename
        self.pet_rhs_filename = pet_rhs_filename
        self.pet_rlds_filename = pet_rlds_filename
        self.pet_rsds_filename = pet_rsds_filename
        self.pet_wind_filename = pet_wind_filename
        self.runoff_params_filename = runoff_params_filename
        self.runoff_tmin_file = runoff_tmin_file
        self.runoff_pr_file = runoff_pr_file
        self.output_file = output_file

    def read_template(self):
        """Read in template ini file."""

        with open(self.template_config_file) as get:
            return get.read()

    def write_template(self, template):
        """Write template to output file."""

        with open(self.output_file, "w") as out:
            out.write(template)

    def spawn(self):
        """Modify template file with replacement values."""

        template = self.read_template()

        template = template.replace(ConfigGenerator.PROJECT_NAME_KEY, self.project_name)
        template = template.replace(ConfigGenerator.ROOT_DIRECTORY_KEY, self.root_directory)
        template = template.replace(ConfigGenerator.START_YEAR_KEY, self.start_year)
        template = template.replace(ConfigGenerator.END_YEAR_KEY, self.end_year)
        template = template.replace(ConfigGenerator.PET_TAS_KEY, self.pet_tas_filename)
        template = template.replace(ConfigGenerator.PET_TMIN_KEY, self.pet_tmin_filename)
        template = template.replace(ConfigGenerator.PET_RHS_KEY, self.pet_rhs_filename)
        template = template.replace(ConfigGenerator.PET_RLDS_KEY, self.pet_rlds_filename)
        template = template.replace(ConfigGenerator.PET_RSDS_KEY, self.pet_rsds_filename)
        template = template.replace(ConfigGenerator.PET_WIND_KEY, self.pet_wind_filename)
        template = template.replace(ConfigGenerator.RUNOFF_PARAMS_KEY, self.runoff_params_filename)
        template = template.replace(ConfigGenerator.RUNOFF_TMIN_KEY, self.runoff_tmin_file)
        template = template.replace(ConfigGenerator.RUNOFF_PR_KEY, self.runoff_pr_file)

        self.write_template(template)


if __name__ == "__main__":

    # task index from SLURM array to run specific scenario, model combinations
    task_id = int(sys.argv[1])

    # number of jobs per node to use for parallel processing; -1 is all
    njobs = 4  # int(sys.argv[2])

    # data directory where climate data directory structure is housed
    data_dir = "/Users/d3y010/projects/climate/mit"  # sys.argv[3]

    # directory to store the outputs in
    output_directory = "/Users/d3y010/projects/xanthos/example/input"  # sys.argv[4]

    # xanthos reference file path with filename and extension
    xanthos_reference_file = "/Users/d3y010/projects/xanthos/example/input/reference/xanthos_0p5deg_landcell_reference.csv"  # sys.argv[5]

    # penman monteith file directory
    pet_data_dir = "/Users/d3y010/projects/xanthos/example/input/pet/penman_monteith"

    # xanthos climate data directory
    xanthos_climate_data_dir = "/Users/d3y010/projects/xanthos/example/input/climate"

    # dict of target variables to extract with TARGET units, not native units; some require conversion in the code
    target_variables = {"FLDS": "w-per-m2",  # surface incident longwave radiation
                        "FSDS": "w-per-m2",  # surface incident shortwave radiation
                        "Hurs": "percent",  # near surface relative humidity
                        "PRECTmmd": "mm-per-month",  # precipitation rate (native units mm/day)
                        "Tair": "degrees-C",  # near surface air temperature (native units K)
                        "Tmin": "degrees-C",  # monthly mean of daily minimum near surface air temperature (native units K)
                        "WIND": "m-per-sec"}  # near surface wind speed

    # scenario name to process; should mirror the associated directory name
    scenario_list = ["BASECOV", "PFCOV", "PARIS_1p5C", "PARIS_2C"]

    # list of model names to process
    model_list = ["ACCESS-ESM1-5", "AWI-ESM-1-1-LR", "BCC-CSM2-MR", "CanESM5",
                  "CMCC-ESM2", "CNRM-ESM2-1", "EC-Earth3-Veg", "FGOALS-g3", "FIO-ESM-2-0",
                  "GISS-E2-2-G", "HadGEM3-GC31-MM", "INM-CM5-0", "IPSL-CM6A-LR", "MIROC-ES2L",
                  "MPI-ESM1-2-HR", "MRI-ESM2-0", "SAM0-UNICON", "UKESM1-0-LL"]

    # create cross product list of scenario, model
    scenario_model_list = [i for i in itertools.product(scenario_list, model_list)]

    # get the scenario, model to process based off of the task id
    scenario, model = scenario_model_list[task_id]

    climate_file = "/Users/d3y010/projects/climate/mit/BASECOV/ACCESS-ESM1-5/2021_2100_0.5_e001_monthly.nc"

    target_output_files = run_extraction(climate_file=climate_file,
                                           xanthos_reference_file=xanthos_reference_file,
                                           target_variables=target_variables,
                                           pet_output_dir=pet_data_dir,
                                           climate_output_dir=xanthos_climate_data_dir,
                                           scenario=scenario,
                                           model=model,
                                           start_year=2021,
                                           through_year=2100,
                                           stitch_to_historic=True)

