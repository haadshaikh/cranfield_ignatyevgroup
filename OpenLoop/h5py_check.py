import h5py

def print_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")

with h5py.File('your_file.h5', 'r') as f:
    f.visititems(print_structure)
