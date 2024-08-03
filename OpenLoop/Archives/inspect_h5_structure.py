# This script inspects the file structure and outputs all the data that is stored within the .h5 data format

import h5py

def print_h5_structure(file_path):
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")

    with h5py.File(file_path, 'r') as f:
        f.visititems(print_attrs)

if __name__ == "__main__":
    # Path to the .data.h5 file
    data_h5_file = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Case -2-AIL-LEFT/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45.data.h5'

    # Print the structure of the HDF5 file
    print_h5_structure(data_h5_file)
