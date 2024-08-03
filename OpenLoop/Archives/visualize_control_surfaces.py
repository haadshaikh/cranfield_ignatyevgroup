import numpy as np
import matplotlib.pyplot as plt

# Example data for visualization
y_coord_ailerons = np.array([0.862823, 2.820273, 4.301239, 5.653424, 6.928342]) / 2.
control_surface_deflections = {
    'aileron_right_inner1': 0,  # Deflection in radians (0.5236 = 30 degrees)
    'aileron_right_inner2': 0,
    'aileron_right_outer1': 0,
    'aileron_right_outer2': 0,
    'aileron_left_inner1': -0,
    'aileron_left_inner2': -0,
    'aileron_left_outer1': -0,
    'aileron_left_outer2': -0
}

def plot_control_surfaces(y_coords, deflections, wing_span):
    fig, ax = plt.subplots(figsize=(10, 6))
    half_wing_span = wing_span / 2

    # Plot wing outline
    ax.plot([-half_wing_span, half_wing_span], [0, 0], 'k-', lw=2, label='Wing')

    # Plot control surfaces with deflections
    num_ailerons = len(y_coords) // 2  # Number of ailerons per side
    for i in range(num_ailerons):
        y = y_coords[i]
        # Right wing
        right_inner_key = f'aileron_right_inner{i + 1}'
        right_outer_key = f'aileron_right_outer{i + 1}'
        if right_inner_key in deflections and right_outer_key in deflections:
            ax.plot([y, y], [0, deflections[right_inner_key]], 'r-', lw=2, label=f'{right_inner_key}' if i == 0 else "")
            ax.plot([y, y], [0, deflections[right_outer_key]], 'b-', lw=2, label=f'{right_outer_key}' if i == 0 else "")

        # Left wing
        y_left = y
        left_inner_key = f'aileron_left_inner{i + 1}'
        left_outer_key = f'aileron_left_outer{i + 1}'
        if left_inner_key in deflections and left_outer_key in deflections:
            ax.plot([-y_left, -y_left], [0, deflections[left_inner_key]], 'r-', lw=2, label=f'{left_inner_key}' if i == 0 else "")
            ax.plot([-y_left, -y_left], [0, deflections[left_outer_key]], 'b-', lw=2, label=f'{left_outer_key}' if i == 0 else "")

    ax.set_xlabel('Spanwise Position (m)')
    ax.set_ylabel('Deflection (rad)')
    ax.legend()
    ax.set_title('Control Surface Deflections')
    plt.grid(True)
    plt.show()

# Wing span example
wing_span = 14.14  # Total wing span (2 * half_wing_span)

# Call the function to plot
plot_control_surfaces(y_coord_ailerons, control_surface_deflections, wing_span)
