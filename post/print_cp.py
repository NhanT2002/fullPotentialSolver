import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

        return data


data = readCGNS("../output/output_384.cgns")


plt.figure()
plt.plot(data['X_wall'], data['Cp_wall'], '-', label='data')
# plt.plot(data_hspm['X_wall'], data_hspm['Cp_wall'], '-', label='HSPM')
plt.gca().invert_yaxis()
plt.xlabel('x')
plt.ylabel('Cp on wall')
plt.title('Pressure Coefficient Distribution on Wall')
plt.legend()
plt.grid()

plt.figure()
plt.semilogy(data['it'], data['res'], label='data')
plt.xlabel('Iteration')
plt.ylabel('Normalized Residual')
plt.title('Convergence History')
plt.legend()
plt.grid()

# print Cp
print("Cp distribution on wall:")
for x, cp in zip(data['X_wall'], data['Cp_wall']):
    print(f"x: {x:.4f}, Cp: {cp:.4f}")