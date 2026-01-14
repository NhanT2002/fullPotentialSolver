import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import io

def readCGNS(filename) :
    # Read CGNS file
    with h5py.File(filename, 'r') as f:

        # x = f['Base/dom-1/GridCoordinates/CoordinateX/ data'][:]
        # y = f['Base/dom-1/GridCoordinates/CoordinateY/ data'][:]
        # z = f['Base/dom-1/GridCoordinates/CoordinateZ/ data'][:]

        # VelocityX = f['Base/dom-1/FLOW_SOLUTION_CC/VelocityX/ data'][:]
        # VelocityY = f['Base/dom-1/FLOW_SOLUTION_CC/VelocityY/ data'][:]
        # VelocityZ = f['Base/dom-1/FLOW_SOLUTION_CC/VelocityZ/ data'][:]

        X = f['WallBase/wall/WALL_FLOW_SOLUTION_CC/xWall/ data'][:]
        Y = f['WallBase/wall/WALL_FLOW_SOLUTION_CC/yWall/ data'][:]
        nx = f['WallBase/wall/WALL_FLOW_SOLUTION_CC/nxWall/ data'][:]
        ny = f['WallBase/wall/WALL_FLOW_SOLUTION_CC/nyWall/ data'][:]
        VelocityX_wall = f['WallBase/wall/WALL_FLOW_SOLUTION_CC/uWall/ data'][:]
        VelocityY_wall = f['WallBase/wall/WALL_FLOW_SOLUTION_CC/vWall/ data'][:]
        Cp = f['WallBase/wall/WALL_FLOW_SOLUTION_CC/cpWall/ data'][:]

        xWake = f['WakeBase/wake/WAKE_FLOW_SOLUTION_NC/xWake/ data'][:]
        yWake = f['WakeBase/wake/WAKE_FLOW_SOLUTION_NC/yWake/ data'][:]
        gammaWake = f['WakeBase/wake/WAKE_FLOW_SOLUTION_NC/gammaWake/ data'][:]

        it = f['Base/GlobalConvergenceHistory/IterationCounters/ data'][:]
        time = f['Base/GlobalConvergenceHistory/Time/ data'][:]
        res = f['Base/GlobalConvergenceHistory/Residual/ data'][:]
        cl = f['Base/GlobalConvergenceHistory/Cl/ data'][:]
        cd = f['Base/GlobalConvergenceHistory/Cd/ data'][:]
        cm = f['Base/GlobalConvergenceHistory/Cm/ data'][:]
        circulation = f['Base/GlobalConvergenceHistory/Circulation/ data'][:]

        # phi = f['Base/dom-1/FLOW_SOLUTION_CC/phi/ data'][:]
        # u = f['Base/dom-1/FLOW_SOLUTION_CC/u/ data'][:]
        # v = f['Base/dom-1/FLOW_SOLUTION_CC/v/ data'][:]
        # rho = f['Base/dom-1/FLOW_SOLUTION_CC/rho/ data'][:]

        # time = np.zeros_like(it)  # Placeholder for time, as it is not available in the file
        # resPhi = np.zeros_like(it)  # Placeholder for resPhi, as it is not available in the file
        # circulation = np.zeros_like(it)  # Placeholder for circulation, as it is not available in the file

        res = res / res[0]  # Normalize residuals

        data = {}

        data['X_wall'] = X
        data['Y_wall'] = Y
        data['nx_wall'] = nx
        data['ny_wall'] = ny
        data['VelocityX_wall'] = VelocityX_wall
        data['VelocityY_wall'] = VelocityY_wall
        data['Cp_wall'] = Cp

        data['it'] = it
        data['time'] = time
        data['res'] = res
        data['cl'] = cl
        data['cd'] = cd
        data['cm'] = cm
        data['circulation'] = circulation

        data['xWake'] = xWake
        data['yWake'] = yWake
        data['gammaWake'] = gammaWake

        return data
    
def unsteadyHistory(filename) :
    # Read CGNS file
    with h5py.File(filename, 'r') as f:

        data = {}

        data['Time'] = f['Base/GlobalConvergenceHistory/Time/ data'][:]
        data['Alpha'] = f['Base/GlobalConvergenceHistory/Alpha/ data'][:]
        data['Cl'] = f['Base/GlobalConvergenceHistory/Cl/ data'][:]
        data['Cd'] = f['Base/GlobalConvergenceHistory/Cd/ data'][:]
        data['Cm'] = f['Base/GlobalConvergenceHistory/Cm/ data'][:]

        data['xWake'] = f['WakeBase/wake/WAKE_FLOW_SOLUTION_NC/xWake/ data'][:]
        data['yWake'] = f['WakeBase/wake/WAKE_FLOW_SOLUTION_NC/yWake/ data'][:]
        data['gammaWake'] = f['WakeBase/wake/WAKE_FLOW_SOLUTION_NC/gammaWake/ data'][:]

        return data


def readHSPM(filename) :
    data = pd.read_csv(filename, sep=' ', skiprows=1, header=None)
    x =  data[0].values
    y =  data[1].values
    z =  data[2].values
    cp = data[3].values

    return {'X_wall': x,
            'Y_wall': y,
            'Cp_wall': cp}

def make_gif_from_circulation(circulation_list, gif_name, duration=100):
    """
    Generate a GIF from a list of circulation (gammaWake) arrays.
    
    Parameters:
    -----------
    circulation_list : list of arrays
        List containing gammaWake data from each timestep
    gif_name : str
        Output filename for the GIF
    duration : int
        Duration of each frame in milliseconds
    """
    images = []
    
    # Find global min/max for consistent y-axis scaling
    all_values = np.concatenate([c.flatten() for c in circulation_list])
    y_min, y_max = np.min(all_values), np.max(all_values)
    # Add some padding
    y_range = y_max - y_min
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    for i, gamma in enumerate(circulation_list):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot circulation distribution
        ax.plot(gamma.flatten(), 'b-', linewidth=1.5)
        
        ax.set_xlabel('Wake Panel Index', fontsize=12)
        ax.set_ylabel('Circulation (γ)', fontsize=12)
        ax.set_title(f'Wake Circulation Distribution - Timestep {i+1}', fontsize=14)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        
        # Convert plot to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(circulation_list)} frames")
    
    # Save as GIF
    images[0].save(
        gif_name,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved as: {gif_name}")



# data_unsteady = unsteadyHistory("../output/output_200.cgns")
# plt.figure()
# plt.plot(data_unsteady['Alpha'], data_unsteady['Cl'], "->",label='Cl')

# plt.figure()
# plt.plot(data_unsteady['Time'], data_unsteady['Alpha'], label='Alpha')

# circulation_list = []
# for i in range(267, 347) :
#     data = unsteadyHistory(f"../output/output_{i}.cgns")
#     circulation_list.append(data['gammaWake'])

# make_gif_from_circulation(
#             circulation_list, 
#             "circulation_animation.gif", 
#             duration=100)


data = readCGNS("../output/output_A0.cgns")
data2 = readCGNS("../output/output_31.cgns")
data3 = readCGNS("../output/output_30.cgns")

data_hspm = readHSPM("../output/HSPM_naca0012_A1-25.dat")


plt.figure()
plt.plot(data['X_wall'], data['Cp_wall'], '-', label='data')
plt.plot(data2['X_wall'], data2['Cp_wall'], '-', label='data2')
plt.plot(data3['X_wall'], data3['Cp_wall'], '-', label='data3')
# plt.plot(data_hspm['X_wall'], data_hspm['Cp_wall'], '-', label='HSPM')
plt.gca().invert_yaxis()
plt.xlabel('x')
plt.ylabel('Cp on wall')
plt.title('Pressure Coefficient Distribution on Wall')
plt.legend()
plt.grid()

plt.figure()
plt.semilogy(data['it'], data['res'], label='data')
plt.semilogy(data2['it'], data2['res'], label='data2')
plt.semilogy(data3['it'], data3['res'], label='data3')
plt.xlabel('Iteration')
plt.ylabel('Normalized Residual')
plt.title('Convergence History')
plt.legend()
plt.grid()

plt.figure()
plt.semilogy(data['time'], data['res'], label='data')
plt.semilogy(data2['time'], data2['res'], label='data2')
plt.semilogy(data3['time'], data3['res'], label='data3')
plt.xlabel('Time')
plt.ylabel('Normalized Residual')
plt.title('Convergence History')
plt.legend()
plt.grid()



# # plot circle of radius 0.5
# theta = np.linspace(0, 2 * np.pi, 100)
# x_circle = 0.5 * np.cos(theta)
# y_circle = 0.5 * np.sin(theta)

# plt.figure()
# # plt.quiver(data['X_wall'], data['Y_wall'], data['nx_wall'], data['ny_wall'])
# plt.quiver(data['X_wall'], data['Y_wall'], data['VelocityX_wall'], data['VelocityY_wall'])
# plt.axis('equal')
# plt.xlim([0.95, 1.05])

# # plt.plot(x_circle, y_circle, 'r--')

# # compute n dot Velocity
# ndotV = data['nx_wall'] * data['VelocityX_wall'] + data['ny_wall'] * data['VelocityY_wall']
# max_ndotV = np.max(np.abs(ndotV))
# print("Max |n dot V| on wall: ", max_ndotV)