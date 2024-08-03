import pandas as pd

# Path to the uploaded CSV file
csv_file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/quat_all_timesteps.csv'

# Load the CSV file
quaternion_df = pd.read_csv(csv_file_path)

# Display the first few rows to verify
print(quaternion_df.head())
