import os


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

    # template config file contained in this repository
    template_config_file = "../data/template.ini"

    # output directory to store generated configuration files in
    output_directory = "/rcfs/projects/gcims/projects/mit_climate/config_files"

    # scenario name to process; should mirror the associated directory name
    scenario_list = ["BASECOV", "PFCOV", "PARIS_1p5C", "PARIS_2C"]

    # list of model names to process
    model_list = ["ACCESS-ESM1-5", "AWI-ESM-1-1-LR", "BCC-CSM2-MR", "CanESM5",
                  "CMCC-ESM2", "CNRM-ESM2-1", "EC-Earth3-Veg", "FGOALS-g3", "FIO-ESM-2-0",
                  "GISS-E2-2-G", "HadGEM3-GC31-MM", "INM-CM5-0", "IPSL-CM6A-LR", "MIROC-ES2L",
                  "MPI-ESM1-2-HR", "MRI-ESM2-0", "SAM0-UNICON", "UKESM1-0-LL"]

    # list of realizations to process
    realization_list = [f"e00{i}" if i < 10 else f"e0{i}" for i in range(1, 50 + 1, 1)]

    for scenario in scenario_list:
        for model in model_list:
            for realization in realization_list:

                basename = f"2021_2100_0.5_{realization}_monthly"
                project_name = f"{scenario}__{model}__{realization}"
                root_dir = "/rcfs/projects/gcims/data/xanthos/mit"
                climate_output_dir = "/rcfs/projects/gcims/data/xanthos/mit/input/climate"
                start_year = 2021
                end_year = 2100
                pet_tas_filename = f"Tair__{scenario}__{model}__{basename}.npy"
                pet_tmin_filename = f"Tmin__{scenario}__{model}__{basename}.npy"
                pet_rhs_filename = f"Hurs__{scenario}__{model}__{basename}.npy"
                pet_rlds_filename = f"FLDS__{scenario}__{model}__{basename}.npy"
                pet_rsds_filename = f"FSDS__{scenario}__{model}__{basename}.npy"
                pet_wind_filename = f"WIND__{scenario}__{model}__{basename}.npy"
                runoff_params_filename = "pars_gswp3_watergap2.npy"
                runoff_tmin_file = os.path.join(climate_output_dir, f"Tmin__{scenario}__{model}__{basename}.npy")
                runoff_pr_file = os.path.join(climate_output_dir, f"PRECTmmd__{scenario}__{model}__{basename}.npy")
                output_file = os.path.join(output_directory, f"{project_name}.ini")

                config_generator = ConfigGenerator(template_config_file=template_config_file,
                                                   project_name=project_name,
                                                   root_directory=root_dir,
                                                   start_year=start_year,
                                                   end_year=end_year,
                                                   pet_tas_filename=pet_tas_filename,
                                                   pet_tmin_filename=pet_tmin_filename,
                                                   pet_rhs_filename=pet_rhs_filename,
                                                   pet_rlds_filename=pet_rlds_filename,
                                                   pet_rsds_filename=pet_rsds_filename,
                                                   pet_wind_filename=pet_wind_filename,
                                                   runoff_params_filename=runoff_params_filename,
                                                   runoff_tmin_file=runoff_tmin_file,
                                                   runoff_pr_file=runoff_pr_file,
                                                   output_file=output_file)

                config_generator.spawn()
