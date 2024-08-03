import h5py
import numpy as np
import matplotlib.pyplot as plt

# Function to read gust parameters from HDF5 file
def read_gust_parameters(hdf5_file_path):
    with h5py.File(hdf5_file_path, 'r') as f:
        gust_parameters = {}
        gust_parameters['gust_shape'] = f['data/settings/StepUvlm/velocity_field_input/gust_shape'][()].decode('utf-8')
        gust_parameters['gust_length'] = float(f['data/settings/StepUvlm/velocity_field_input/gust_parameters/gust_length'][()])
        gust_parameters['gust_intensity'] = float(f['data/settings/StepUvlm/velocity_field_input/gust_parameters/gust_intensity'][()])
        gust_parameters['gust_offset'] = float(f['data/settings/StepUvlm/velocity_field_input/offset'][()])
        gust_parameters['relative_motion'] = f['data/settings/StepUvlm/velocity_field_input/relative_motion'][()].decode('utf-8')

    return gust_parameters

# Function to generate gust profile
def generate_gust_profile(gust_parameters, total_timesteps=250, total_duration=0.6542):
    gust_shape = gust_parameters['gust_shape']
    gust_length = gust_parameters['gust_length']
    gust_intensity = gust_parameters['gust_intensity']
    gust_offset = gust_parameters['gust_offset']
    
    # Calculate the time step
    time_step = total_duration / total_timesteps

    # Time settings
    time = np.linspace(0, total_duration, total_timesteps)

    # Initialize gust profile
    gust_profile = np.zeros_like(time)

    # Calculate the gust profile
    start_index = int(gust_offset / time_step)
    end_index = int((gust_offset + gust_length) / time_step)

    # Ensure indices are within bounds
    start_index = min(start_index, total_timesteps)
    end_index = min(end_index, total_timesteps)

    if gust_shape == 'lateral 1-cos':
        for i in range(start_index, end_index):
            t = (i - start_index) * time_step
            gust_profile[i] = 0.5 * gust_intensity * (1 - np.cos(2 * np.pi * t / gust_length))

    return time, gust_profile

# Function to plot gust profile
def plot_gust_profile(time, gust_profile, gust_offset, gust_length, gust_intensity):
    plt.figure(figsize=(10, 6))
    plt.plot(time, gust_profile, label='Gust Profile', linewidth=2)
    plt.axvline(gust_offset, color='r', linestyle='--', label='Gust Start')
    plt.axvline(gust_offset + gust_length, color='g', linestyle='--', label='Gust End')
    plt.xlabel('Time [s]')
    plt.ylabel('Gust Intensity')
    plt.title(f'Gust Profile (Lateral 1-cos) - Intensity: {gust_intensity}, Length: {gust_length}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Path to the HDF5 file
hdf5_file_path = '/home/haadshaikh/Desktop/Research_Projects/IRP/IRP_Flexop_Simulations/OpenLoop/output/output backups/Lateral_Gust_Case/CASE-1-No-Input/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45/savedata/flexop_free_gust_L_10_I_10_p_0_cfl_0_uinf45.data.h5'

# Read gust parameters from HDF5 file
gust_parameters = read_gust_parameters(hdf5_file_path)

# Print gust parameters to verify
print("Gust Parameters:", gust_parameters)

# Generate gust profile
time, gust_profile = generate_gust_profile(gust_parameters)

# Plot gust profile
plot_gust_profile(time, gust_profile, gust_parameters['gust_offset'], gust_parameters['gust_length'], gust_parameters['gust_intensity'])


'''

Plotting a Simple Gust Profile



import numpy as np
import matplotlib.pyplot as plt

# Gust settings
gust_shape = 'lateral 1-cos'
gust_length = 10.0  # Duration of the gust in seconds
gust_intensity = 0.1  # Peak intensity of the gust
gust_offset = 10  # Offset at which the gust starts in seconds

# Time settings
total_duration = 30  # Total duration of the simulation in seconds
time_step = 0.1  # Time step for the simulation in seconds
time = np.arange(0, total_duration, time_step)

# Initialize gust profile
gust_profile = np.zeros_like(time)

# Calculate the gust profile
start_index = int(gust_offset / time_step)
end_index = int((gust_offset + gust_length) / time_step)

for i in range(start_index, end_index):
    t = (i - start_index) * time_step
    gust_profile[i] = 0.5 * gust_intensity * (1 - np.cos(2 * np.pi * t / gust_length))

# Plot the gust profile
plt.figure(figsize=(10, 6))
plt.plot(time, gust_profile, label='Gust Profile', linewidth=2)
plt.axvline(gust_offset, color='r', linestyle='--', label='Gust Start')
plt.axvline(gust_offset + gust_length, color='g', linestyle='--', label='Gust End')
plt.xlabel('Time [s]')
plt.ylabel('Gust Intensity')
plt.title('Gust Profile (Lateral 1-cos)')
plt.legend()
plt.grid(True)
plt.show()
'''