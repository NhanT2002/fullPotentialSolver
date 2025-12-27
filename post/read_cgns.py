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

def readHSPM(filename) :
    data = pd.read_csv(filename, sep=' ', skiprows=1, header=None)
    x =  data[0].values
    y =  data[1].values
    z =  data[2].values
    cp = data[3].values

    return {'X_wall': x,
            'Y_wall': y,
            'Cp_wall': cp}

data = readCGNS("../output/output_230.cgns")
data2 = readCGNS("../output/output_231.cgns")
data3 = readCGNS("../output/output_237.cgns")

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