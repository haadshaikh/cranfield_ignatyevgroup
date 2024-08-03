# New Gamma Extractor script

import h5py
import numpy as np

class GammaExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_gamma(self):
        with h5py.File(self.file_path, 'r') as hdf:
            # Iterate through timesteps and attempt to extract gamma values
            gamma_values = {}
            timesteps = list(hdf['data/structure/timestep_info'].keys())
            for t in timesteps:
                try:
                    gamma = hdf[f'data/structure/timestep_info/{t}/postproc_node/aero_unsteady_forces'][:]
                    gamma_values[t] = gamma
                except KeyError:
                    print(f"Gamma values not found for timestep {t}")
                    continue
        return gamma_values

if __name__ == "__main__":
    file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45.data.h5'
    extractor = GammaExtractor(file_path)
    gamma_values = extractor.extract_gamma()
    print("Extracted gamma values for each timestep:")
    for timestep, gamma in gamma_values.items():
        print(f"Timestep {timestep}: Gamma values shape: {gamma.shape}")


'''
import os
import h5py as h5
import numpy as np
import pandas as pd

def extract_gamma(h5_file_path, csv_file_path):
    # Open the HDF5 file
    with h5.File(h5_file_path, 'r') as f:
        # Navigate to the correct group and dataset
        aero_group = f['data']['structure']['timestep_info']
        
        # Initialize an empty list to store the gamma data
        gamma_data = []
        
        # Loop through each time step to extract gamma
        for ts in aero_group.keys():
            if ts.startswith('_'):
                continue
            ts_group = aero_group[ts]
            postproc_node = ts_group['postproc_node']
            
            # Check if 'gamma' dataset exists
            if 'gamma' in postproc_node.keys():
                gamma = postproc_node['gamma']
                gamma_data.append(gamma[:])
            else:
                raise KeyError("The dataset 'gamma' is not found in 'postproc_node' group")
        
        # Convert the list of gamma arrays into a single numpy array
        gamma_data = np.vstack(gamma_data)
        
        # Save the gamma data to a CSV file
        pd.DataFrame(gamma_data).to_csv(csv_file_path, index=False, header=False)

# Set the file paths
h5_file_path = "/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45.data.h5"
csv_file_path = "/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/gamma.csv"

# Run the function
extract_gamma(h5_file_path, csv_file_path)
'''