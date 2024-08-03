# Import necessary modules
import sys
sys.path.insert(1, '../../01_Case_Generator')
from generate_flexop_case import generate_flexop_case

# Define simulation parameters
u_inf = 45  # Cruise flight speed in m/s
rho = 1.1336  # Air density in kg/m^3 (corresponding to an altitude of 800m)
alpha_rad = 6.406771329255241468e-03  # Angle of attack in radians (approximately 0.389 degrees)

# Simulation settings
simulation_settings = {
    'lifting_only': True,
    'wing_only': False,
    'dynamic': True,
    'wake_discretisation': True,
    'gravity': True,
    'horseshoe': False,
    'use_polars': False,
    'free_flight': True,    # This was false
    'use_trim': False,
    'mstar': 40,            # Changed from 80
    'num_chord_panels': 4,  # Changed from 8
    'n_elem_multiplier': 1, # Changed from 2
    'n_tstep': 9550,
    'num_cores': 8,
    'sigma': 0.3,
    'postprocessors_dynamic': ['BeamLoads', 'SaveData', 'BeamPlot', 'AerogridPlot'],
}

# Initial trim values
initial_trim_values = {
    'alpha': alpha_rad,
    'delta': -3.325087601649625961e-03,
    'thrust': 3.0,     # Previously 2.052055145318664842e+00
    'cs_deflections': {
        'aileron_right_inner1': 0.25,  # Set to zero for testing
        'aileron_right_inner2': 0.25,
        'aileron_right_outer1': 0.25,
        'aileron_right_outer2': 0.25,
        'aileron_left_inner1': -0.25,
        'aileron_left_inner2': -0.25,
        'aileron_left_outer1': -0.25,
        'aileron_left_outer2': -0.25
    }
}

# Gust settings
gust_settings = {
    'use_gust': False,              # Changed from True
    'gust_shape': 'lateral 1-cos',  # Changed from 1-cos
    'gust_length': 10.0,            # Changed from 10.0
    'gust_intensity': 0.1,          # Changed from 0.1
    'gust_offset': 10
}

# Set wake shape inputs if needed for variable wake discretization
if simulation_settings['wake_discretisation']:
    dict_wake_shape = {
        'dx1': 0.471 / simulation_settings['num_chord_panels'],
        'ndx1': 23,
        'r': 1.6,
        'dxmax': 5 * 0.471
    }
    simulation_settings['mstar'] = 35
    print(simulation_settings)
else:
    dict_wake_shape = None

# Define the flow sequence
flow = [
    'BeamLoader',
    'AerogridLoader',
    'NonliftingbodygridLoader',
    'AerogridPlot',
    'BeamPlot',
    'StaticCoupled',
    'DynamicCoupled'
]




# Remove certain steps based on simulation settings
if simulation_settings['lifting_only']:
    flow.remove('NonliftingbodygridLoader')

# Loop over various gust lengths
list_gust_lengths = [10]  # List of gust lengths to simulate

for gust_length in list_gust_lengths:
    gust_settings['gust_length'] = gust_length

    # Generate a case name based on simulation settings
    case_name = 'flexop_free_gust_L_{}_I_{}_p_{}_cfl_{}_uinf{}'.format(
        gust_settings['gust_length'],
        int(gust_settings['gust_intensity'] * 100),
        int(simulation_settings['use_polars']),
        int(not simulation_settings['wake_discretisation']),
        int(u_inf)
    )

    if not simulation_settings["lifting_only"]:
        case_name += '_nonlifting'

    # Generate the FlexOP model and start the simulation
    cs_deflections = initial_trim_values.get('cs_deflections', {})
    flexop_model = generate_flexop_case(
        u_inf,
        rho,
        flow,
        initial_trim_values,
        case_name,
        gust_settings=gust_settings,
        dict_wake_shape=dict_wake_shape,
        cs_deflections=cs_deflections,
        **simulation_settings,
        nonlifting_interactions=bool(not simulation_settings["lifting_only"])
    )
    flexop_model.run()


'''# Import necessary modules
import sys
sys.path.insert(1, '../../01_Case_Generator')
from generate_flexop_case import generate_flexop_case

# Define simulation parameters
u_inf = 45  # Cruise flight speed in m/s
rho = 1.1336  # Air density in kg/m^3 (corresponding to an altitude of 800m)
alpha_rad = 6.406771329255241468e-03  # Angle of attack in radians (approximately 0.389 degrees)

# Simulation settings
simulation_settings = {
    'lifting_only': True,
    'wing_only': False,
    'dynamic': True,
    'wake_discretisation': True,
    'gravity': True,
    'horseshoe': False,
    'use_polars': False,
    'free_flight': True,    # This was false
    'use_trim': False,
    'mstar': 40,            # Changed from 80
    'num_chord_panels': 4,  # Changed from 8
    'n_elem_multiplier': 1, # Changed from 2
    'n_tstep': 200,
    'num_cores': 8,
    'sigma': 0.3,
    'postprocessors_dynamic': ['BeamLoads', 'SaveData', 'BeamPlot', 'AerogridPlot'],
}

# Initial trim values
initial_trim_values = {
    'alpha': alpha_rad,
    'delta': -3.325087601649625961e-03,
    'thrust': 3.0,     # Previously 2.052055145318664842e+00
    'cs_deflections': {
        'aileron_right_inner1': 0.25,  # Set to zero for testing
        'aileron_right_inner2': 0.25,
        'aileron_right_outer1': 0.25,
        'aileron_right_outer2': 0.25,
        'aileron_left_inner1': -0.25,
        'aileron_left_inner2': -0.25,
        'aileron_left_outer1': -0.25,
        'aileron_left_outer2': -0.25
    }
}

# Gust settings
gust_settings = {
    'use_gust': True,
    'gust_shape': '1-cos',
    'gust_length': 10.0,
    'gust_intensity': 0.1,
    'gust_offset': 10
}

# Set wake shape inputs if needed for variable wake discretization
if simulation_settings['wake_discretisation']:
    dict_wake_shape = {
        'dx1': 0.471 / simulation_settings['num_chord_panels'],
        'ndx1': 23,
        'r': 1.6,
        'dxmax': 5 * 0.471
    }
    simulation_settings['mstar'] = 35
    print(simulation_settings)
else:
    dict_wake_shape = None

# Define the flow sequence
flow = [
    'BeamLoader',
    'AerogridLoader',
    'NonliftingbodygridLoader',
    'AerogridPlot',
    'BeamPlot',
    'StaticCoupled',
    'DynamicCoupled'
]

# Remove certain steps based on simulation settings
if simulation_settings['lifting_only']:
    flow.remove('NonliftingbodygridLoader')

# Loop over various gust lengths
list_gust_lengths = [10]  # List of gust lengths to simulate

for gust_length in list_gust_lengths:
    gust_settings['gust_length'] = gust_length

    # Generate a case name based on simulation settings
    case_name = 'flexop_free_gust_L_{}_I_{}_p_{}_cfl_{}_uinf{}'.format(
        gust_settings['gust_length'],
        int(gust_settings['gust_intensity'] * 100),
        int(simulation_settings['use_polars']),
        int(not simulation_settings['wake_discretisation']),
        int(u_inf)
    )

    if not simulation_settings["lifting_only"]:
        case_name += '_nonlifting'

    # Generate the FlexOP model and start the simulation
    cs_deflections = initial_trim_values.get('cs_deflections', {})
    flexop_model = generate_flexop_case(
        u_inf,
        rho,
        flow,
        initial_trim_values,
        case_name,
        gust_settings=gust_settings,
        dict_wake_shape=dict_wake_shape,
        cs_deflections=cs_deflections,
        **simulation_settings,
        nonlifting_interactions=bool(not simulation_settings["lifting_only"])
    )
    flexop_model.run()
'''
