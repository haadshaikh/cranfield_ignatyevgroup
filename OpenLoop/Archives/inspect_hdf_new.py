import h5py

def list_datasets(file_path, output_file):
    datasets = []
    def collect_attrs(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)
    
    with h5py.File(file_path, 'r') as hdf:
        hdf.visititems(collect_attrs)
    
    with open(output_file, 'w') as f:
        for dataset in datasets:
            f.write(f"{dataset}\n")

file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Case -2-AIL-LEFT/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45.data.h5'
output_file = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/hdf5_datasets.txt'

list_datasets(file_path, output_file)
