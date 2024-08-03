import h5py
import pandas as pd
import numpy as np

# Sample HDF5 file path
file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-1-No-Input/Longer_Timestep_1500/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45.data.h5'

# List of timesteps to check
timesteps = list(range(1501))

# Initialize dictionaries to hold the data for each parameter
data_dicts = {
    'gamma_00000': [],
    'gamma_00001': [],
    'for_pos': [],
    'for_vel': [],
    'loads': [],
    'u_inf': [],
    'strain': [],
    'incidence_angle': [],
    'quat': [],
    'u_ext': [],
}

# Open the HDF5 file
with h5py.File(file_path, 'r') as f:
    for timestep in timesteps:
        timestep_str = f"{timestep:05d}"

        # Define paths
        paths = {
            'gamma_00000': f"data/aero/timestep_info/{timestep_str}/gamma/00000",
            'gamma_00001': f"data/aero/timestep_info/{timestep_str}/gamma/00001",
            'for_pos': f"data/structure/timestep_info/{timestep_str}/for_pos",
            'for_vel': f"data/structure/timestep_info/{timestep_str}/for_vel",
            'loads': f"data/structure/timestep_info/{timestep_str}/postproc_cell/loads",
            'u_inf': f"data/settings/DynamicCoupled/aero_solver_settings/velocity_field_input/u_inf",
            'strain': f"data/structure/timestep_info/{timestep_str}/postproc_cell/strain",
            'incidence_angle': f"data/settings/AerogridPlot/include_incidence_angle",
            'quat': f"data/structure/timestep_info/{timestep_str}/quat",
            'u_ext': f"data/aero/timestep_info/00000/u_ext/00000"
        }

        # Extract data
        for name, path in paths.items():
            try:
                data = f[path][()]
                if isinstance(data, np.ndarray):
                    data_dicts[name].append((timestep, *data.flatten()))
                else:
                    data_dicts[name].append((timestep, data))
            except KeyError:
                print(f"{name} data not found for timestep {timestep} at {path}")

# Save each parameter's data to a separate CSV
for name, data_list in data_dicts.items():
    if data_list:
        columns = ['timestep'] + [f"{name}_{i}" for i in range(len(data_list[0]) - 1)]
        df = pd.DataFrame(data_list, columns=columns)
        df.to_csv(f"{name}_all_timesteps.csv", index=False)

# Optionally, combine all data into a single CSV
combined_df = pd.concat([pd.DataFrame(data_dicts[name], columns=['timestep'] + [f"{name}_{i}" for i in range(len(data_dicts[name][0]) - 1)]) for name in data_dicts if data_dicts[name]], axis=1)
combined_df.to_csv("combined_all_parameters.csv", index=False)
