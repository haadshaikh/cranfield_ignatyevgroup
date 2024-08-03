import numpy as np
import pandas as pd
from scipy.linalg import svd, pinv

def load_simulation_data(states_file_path, inputs_file_path, outputs_file_path):
    # Load the data from the ODS files without headers
    states_data = pd.read_excel(states_file_path, engine='odf', header=None)
    inputs_data = pd.read_excel(inputs_file_path, engine='odf', header=None)
    outputs_data = pd.read_excel(outputs_file_path, engine='odf', header=None)

    # Convert state data to numpy array and transpose to get (variables, time steps)
    state_matrix = states_data.to_numpy().T

    # Convert control inputs to numpy array and transpose to get (controls, time steps)
    control_matrix = inputs_data.to_numpy().T
    
    # Convert output data to numpy array and transpose to get (outputs, time steps)
    output_matrix = outputs_data.to_numpy().T

    return state_matrix, control_matrix, output_matrix

def check_for_nan_inf(matrix, matrix_name):
    if np.isnan(matrix).any() or np.isinf(matrix).any():
        print(f"{matrix_name} contains nan or inf values.")
        indices = np.where(np.isnan(matrix) | np.isinf(matrix))
        print(f"indices with nan or inf in {matrix_name}: {indices}")
        return True, indices
    return False, None

def clean_data(X1, X2, U):
    # Find indices with nan or inf in X2
    _, nan_inf_indices_X2 = check_for_nan_inf(X2, "X2_combined")

    if nan_inf_indices_X2 is not None:
        # Remove columns with nan or inf values in X2
        valid_indices_X2 = np.ones(X2.shape[1], dtype=bool)
        valid_indices_X2[nan_inf_indices_X2[1]] = False

        # Ensure U has one less column than X1 and X2
        valid_indices_U = np.ones(U.shape[1], dtype=bool)
        valid_indices_U[nan_inf_indices_X2[1][:-1]] = False

        X1_clean = X1[:, valid_indices_X2]
        X2_clean = X2[:, valid_indices_X2]
        U_clean = U[:, valid_indices_U]

        return X1_clean, X2_clean, U_clean

    return X1, X2, U

def perform_dmdc(X1, X2, U, regularization=1e-8):
    print("Dimensions of X1:", X1.shape)
    print("Dimensions of X2:", X2.shape)
    print("Dimensions of U:", U.shape)

    # Check for nan or inf values in the input matrices
    nan_inf_X1, _ = check_for_nan_inf(X1, "X1_combined")
    nan_inf_X2, indices_X2 = check_for_nan_inf(X2, "X2_combined")
    nan_inf_U, _ = check_for_nan_inf(U, "U_combined")

    if nan_inf_X2:
        X1, X2, U = clean_data(X1, X2, U)
        print("Cleaned data dimensions:")
        print("X1:", X1.shape)
        print("X2:", X2.shape)
        print("U:", U.shape)

    # Perform SVD on the state matrix X1
    U_svd, S_svd, V_svd = svd(X1, full_matrices=False)

    # Regularize the singular values
    S_inv = np.diag(1 / (S_svd + regularization))

    # Compute A
    A = X2 @ V_svd.T @ S_inv @ U_svd.T

    # Compute B
    B = X2 @ U.T @ pinv(U @ U.T + regularization * np.eye(U.shape[0]))

    return A, B

def estimate_C_D(X, U, Y, regularization=1e-8):
    Z = np.vstack([X, U])
    Z_pinv = pinv(Z.T @ Z + regularization * np.eye(Z.shape[0])) @ Z.T
    CD = Y @ Z_pinv
    n_y = Y.shape[0]
    n_x = X.shape[0]
    C = CD[:, :n_x]
    D = CD[:, n_x:]
    return C, D

# List of file paths for all simulations
states_file_paths = [
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-1-No-Input/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-2-FULL-LEFT/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-3-FULL-RIGHT/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-4-OB-LEFT/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-5-IB-LEFT/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-6-OB-RIGHT/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-7-IB-RIGHT/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-8-SURFACE-UP/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-9-SURFACE-DOWN/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-1-No-Input/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-2-FULL-LEFT/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-3-FULL-RIGHT/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-4-OB-LEFT/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-5-IB-LEFT/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-6-OB-RIGHT/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-7-IB-RIGHT/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-8-SURFACE-UP/States.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-9-SURFACE-DOWN/States.ods',    
]

inputs_file_paths = [
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-1-No-Input/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-2-FULL-LEFT/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-3-FULL-RIGHT/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-4-OB-LEFT/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-5-IB-LEFT/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-6-OB-RIGHT/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-7-IB-RIGHT/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-8-SURFACE-UP/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-9-SURFACE-DOWN/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-1-No-Input/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-2-FULL-LEFT/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-3-FULL-RIGHT/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-4-OB-LEFT/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-5-IB-LEFT/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-6-OB-RIGHT/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-7-IB-RIGHT/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-8-SURFACE-UP/Inputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-9-SURFACE-DOWN/Inputs.ods',  
]

outputs_file_paths = [
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-1-No-Input/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-2-FULL-LEFT/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-3-FULL-RIGHT/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-4-OB-LEFT/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-5-IB-LEFT/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-6-OB-RIGHT/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-7-IB-RIGHT/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-8-SURFACE-UP/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-9-SURFACE-DOWN/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-1-No-Input/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-2-FULL-LEFT/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-3-FULL-RIGHT/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-4-OB-LEFT/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-5-IB-LEFT/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-6-OB-RIGHT/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-7-IB-RIGHT/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-8-SURFACE-UP/Outputs.ods',
    '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/No_Gust_Case/CASE-9-SURFACE-DOWN/Outputs.ods',  
]

# Initialize lists to collect data
X1_list = []
X2_list = []
U_list = []
Y_list = []

# Load and concatenate data from all simulations
for states_file_path, inputs_file_path, outputs_file_path in zip(states_file_paths, inputs_file_paths, outputs_file_paths):
    state_matrix, control_matrix, output_matrix = load_simulation_data(states_file_path, inputs_file_path, outputs_file_path)

    # Separate into X1 and X2
    X1 = state_matrix[:, :-1]
    X2 = state_matrix[:, 1:]

    X1_list.append(X1)
    X2_list.append(X2)
    U_list.append(control_matrix[:, :-1])  # Ensure control data matches the state data time steps
    Y_list.append(output_matrix[:, :-1])  # Ensure output data matches the state data time steps

# Concatenate state matrices horizontally
X1_combined = np.hstack(X1_list)
X2_combined = np.hstack(X2_list)
U_combined = np.hstack(U_list)
Y_combined = np.hstack(Y_list)

# Print dimensions for debugging
print("Final dimensions of X1_combined:", X1_combined.shape)
print("Final dimensions of X2_combined:", X2_combined.shape)
print("Final dimensions of U_combined:", U_combined.shape)
print("Final dimensions of Y_combined:", Y_combined.shape)

# Perform DMDc analysis
A, B = perform_dmdc(X1_combined, X2_combined, U_combined)

print("System dynamics matrix A:")
print(pd.DataFrame(A))
print("Control matrix B:")
print(pd.DataFrame(B))

# Estimate C and D matrices
C, D = estimate_C_D(X1_combined, U_combined, Y_combined)

print("Output matrix C:")
print(pd.DataFrame(C))
print("Feedforward matrix D:")
print(pd.DataFrame(D))
