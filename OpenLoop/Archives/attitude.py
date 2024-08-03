import h5py
import pandas as pd

def extract_attitude(file_path):
    with h5py.File(file_path, 'r') as hdf:
        # List all keys to find potential attitude-related data
        keys = list(hdf.keys())
        for key in keys:
            print(key)
        
        # Assuming attitude data might be in a dataset named 'quat' (quaternions) or 'euler' (Euler angles)
        attitude_data = []
        for timestep in range(159, 201):  # Adjust range based on available timesteps
            try:
                quat_key = f'data/structure/timestep_info/{str(timestep).zfill(5)}/quat'
                if quat_key in hdf:
                    quat_data = hdf[quat_key][:]
                    attitude_data.append({'timestep': timestep, 'quat': quat_data.tolist()})
                else:
                    print(f"Skipping {quat_key}: Not found in the HDF5 file.")
            except Exception as e:
                print(f"Error accessing {quat_key}: {e}")
        
        if not attitude_data:
            print("No attitude data found.")
            return None
        
        # Convert to DataFrame
        attitude_df = pd.DataFrame(attitude_data)
        return attitude_df

file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45.data.h5'
attitude_df = extract_attitude(file_path)

if attitude_df is not None:
    # Save to CSV
    attitude_df.to_csv('/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/attitude_data.csv', index=False)
    print("Attitude data has been saved to /mnt/data/attitude_data.csv")
    display_dataframe_to_user(attitude_df)
else:
    print("No attitude data extracted.")