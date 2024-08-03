# This scripts helps with extracting gamma from the .h5 file data structure
# The scipt acquires gamma from 'data/aero/timestep_info/00000/gamma/00000' for each time step and each wing / wake region

import h5py
import pandas as pd

def extract_gamma(file_path):
    with h5py.File(file_path, 'r') as hdf:
        gamma_data = []
        
        # Iterate over all timesteps
        for timestep in range(0, 250):
            gamma_key_prefix = f'data/aero/timestep_info/{timestep:05d}/gamma'
            timestep_data = []
            
            # Iterate over gamma sub-datasets within each timestep
            for i in range(4):  # Assuming there are 4 sub-datasets based on the provided example
                gamma_key = f'{gamma_key_prefix}/{i:05d}'
                if gamma_key in hdf:
                    gamma = hdf[gamma_key][:]
                    gamma_shape = gamma.shape
                    timestep_data.append({
                        'timestep': timestep,
                        'surface': i,
                        'gamma': gamma.tolist(),  # Convert numpy array to list
                        'shape': gamma_shape
                    })
                else:
                    print(f"Skipping {gamma_key}: Not a valid dataset")
            
            # Add the timestep data to the gamma_data list
            if timestep_data:
                gamma_data.extend(timestep_data)

        # Convert to a DataFrame
        gamma_df = pd.DataFrame(gamma_data)
        
        # Save to CSV
        csv_file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/gamma_values_all_timesteps.csv'
        gamma_df.to_csv(csv_file_path, index=False)
        print(f'Gamma values have been saved to {csv_file_path}')

file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Case-2-AIL-LEFT/Sim 7 - Final Case Ail Left/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45.data.h5'
extract_gamma(file_path)
