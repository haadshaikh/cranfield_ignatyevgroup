# Inspect HDF

import h5py

# Define the file path
file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45.data.h5'

# Function to print the structure of the HDF5 file
def inspect_h5_file(file_path):
    with h5py.File(file_path, 'r') as hdf:
        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")
        
        hdf.visititems(print_attrs)

# Call the function to inspect the file
inspect_h5_file(file_path)
