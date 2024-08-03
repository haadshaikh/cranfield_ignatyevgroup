import h5py

# Path to your HDF5 file
hdf5_file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/Case-1-No-Input/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45.data.h5'

# Location in the HDF5 file for the loads data
hdf5_data_path = 'data/structure/timestep_info/00001/postproc_cell/loads'

# Load the HDF5 file and find the shape of the loads parameter
with h5py.File(hdf5_file_path, 'r') as f:
    # Access the dataset
    loads_data = f[hdf5_data_path]
    
    # Find the shape
    loads_shape = loads_data.shape
    print('Shape of the loads parameter:', loads_shape)
