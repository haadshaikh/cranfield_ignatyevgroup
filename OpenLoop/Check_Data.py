# This script checks if the data is present

import h5py

# Sample HDF5 file path
file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Case-1-No-Input/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45.data.h5'

# List of timesteps to check
timesteps = list(range(251))

# Open the HDF5 file
with h5py.File(file_path, 'r') as f:
    for timestep in timesteps:
        timestep_str = f"{timestep:05d}"

        # Define paths
        paths = {
            'gamma_00000': f"data/aero/timestep_info/{timestep_str}/gamma/00000",
            'gamma_00001': f"data/aero/timestep_info/{timestep_str}/gamma/00001",
            'pos': f"data/structure/timestep_info/{timestep_str}/pos",
            'pos_dot': f"data/structure/timestep_info/{timestep_str}/pos_dot",
            'pos_ddot': f"data/structure/timestep_info/{timestep_str}/pos_ddot",
            'control_surface_deflection': f"data/aero/timestep_info/{timestep_str}/control_surface_deflection"
        }

        # Check and print the presence of each path
        for name, path in paths.items():
            try:
                data = f[path][()]
                print(f"{name} data found for timestep {timestep} at {path}")
            except KeyError:
                print(f"{name} data not found for timestep {timestep} at {path}")




'''
# This Gamma Checker Script Works!

import h5py

# Sample HDF5 file path
file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Case-1-No-Input/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45.data.h5'

# List of timesteps to check
timesteps = [158, 159, 160, 161, 162, 163, 164, 165]

# Open the HDF5 file
with h5py.File(file_path, 'r') as f:
    for timestep in timesteps:
        gamma_path = f"data/aero/timestep_info/{timestep:05d}/gamma"
        if gamma_path in f:
            print(f"Gamma data found for timestep {timestep} at {gamma_path}")
        else:
            print(f"Gamma data not found for timestep {timestep} at {gamma_path}")


'''