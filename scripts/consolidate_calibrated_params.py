import os
import glob

import numpy as np


if __name__ == "__main__":

    n_basins = 235
    n_params = 5

    # directory where the calibration outputs were written
    calibration_output_directory = "/rcfs/projects/gcims/models/xanthos/calibration_gswp3_watergap2/data/outputs/gswp3_20220717"

    # file name to write consolidated output file to
    output_calibration_file = "/rcfs/projects/gcims/data/xanthos/mit/input/runoff/abcd/pars_gswp3_watergap2.npy"

    # get a list of individual parameter files to process
    param_files = glob.glob(os.path.join(calibration_output_directory, "abcdm_parameters_popsize-1500_basin-*.npy"))

    for basin_id in range(1, n_basins+1, 1):

        target_file = os.path.join(calibration_output_directory, f"abcdm_parameters_popsize-1500_basin-{basin_id}.npy")

        if basin_id == 1:

            # load the target parameter file
            arr = np.load(target_file)

        else:

            arx = np.load(target_file)

            # concatenate arrays
            arr = np.concatenate([arr, arx])

    # reshape to desired size
    arr = arr.reshape((n_basins, n_params))

    # write output
    np.save(output_calibration_file, arr)
