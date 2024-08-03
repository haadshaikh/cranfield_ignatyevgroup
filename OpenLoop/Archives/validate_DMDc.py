# System Dynamics Simulation to Validate DMDc

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Load A and B matrices from CSV files
A = pd.read_csv('/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/System_Dynamics/Initial_DMDc_Results/A_matrix.csv', header=None).values
B = pd.read_csv('/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/System_Dynamics/Initial_DMDc_Results/B_matrix.csv', header=None).values

# Define C and D matrices
n = A.shape[0]  # Number of states, assuming A is square
m = B.shape[1]  # Number of inputs
C = np.eye(n)   # Identity matrix of shape (n x n)
D = np.zeros((n, m))  # Zero matrix of shape (n x m)

# Define the system dynamics
def system_dynamics(t, x, u):
    dxdt = A @ x + B @ u
    return dxdt

# Initial conditions
x0 = np.zeros(n)  # Initial state vector

# Time span for the simulation
t_span = (0, 1)  # Simulate from t=0 to t=10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 300)  # Time points at which to store the results

# Define input function u(t)
def input_function(t):
    # Define the input as needed, here it's a constant zero input
    input = [-0.05,-0.05,-0.05,-0.05,0.05,0.05,0.05,0.05] # RW IB1,IB2,OB1,OB2, LW IB1,IB2,OB1,OB2
    return input                                          # -ve is surface up | +ve is surface down

# Define a wrapper function for solve_ivp that incorporates time-varying input
def system_dynamics_with_input(t, x):
    u = input_function(t)
    return system_dynamics(t, x, u)

# Solve the system
sol = solve_ivp(system_dynamics_with_input, t_span, x0, t_eval=t_eval, method='RK45')

# Calculate outputs
# Ensure D @ u is computed correctly over time
U = np.array([input_function(t) for t in sol.t]).T
Y = C @ sol.y + D @ U

# Suppose you choose to plot outputs 1, 3, and 5
output_indices = [9, 10, 11]  # Adjust these indices based on your specific parameters of interest

# Create a figure and define the size
plt.figure(figsize=(10, 12))

# Plotting Output 1
plt.subplot(3, 1, 1)  # This creates the first subplot in a grid of 3 rows x 1 column
plt.plot(sol.t, Y[output_indices[0]], label='Vx')
plt.title('Vx')
plt.xlabel('Time (s)')
plt.ylabel('Vx')
plt.grid(True)
plt.legend()

# Plotting Output 3
plt.subplot(3, 1, 2)  # This creates the second subplot
plt.plot(sol.t, Y[output_indices[1]], label='Vy')
plt.title('Vy')
plt.xlabel('Time (s)')
plt.ylabel('Vy')
plt.grid(True)
plt.legend()

# Plotting Output 5
plt.subplot(3, 1, 3)  # This creates the third subplot
plt.plot(sol.t, Y[output_indices[2]], label='Vz')
plt.title('Vz')
plt.xlabel('Time (s)')
plt.ylabel('Vz')
plt.grid(True)
plt.legend()

# Display the plots
plt.tight_layout()  # This helps to ensure the plots are not overlapping
plt.show()