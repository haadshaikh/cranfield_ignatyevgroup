import pandas as pd
from scipy.io import savemat

# Step 1: Read the CSV file using pandas
csv_file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/System_Dynamics/Initial_DMDc_Results/B_matrix.csv'
data_frame = pd.read_csv(csv_file_path)

# Step 2: Convert the DataFrame to a dictionary
data_dict = {col: data_frame[col].values for col in data_frame.columns}

# Step 3: Save the dictionary to a MAT file
mat_file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/System_Dynamics/Initial_DMDc_Results/B_matrix.mat'
savemat(mat_file_path, data_dict)

print(f"CSV file has been successfully converted to MAT file and saved at {mat_file_path}")
