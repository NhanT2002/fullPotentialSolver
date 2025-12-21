import h5py
import json
import matplotlib.pyplot as plt
import numpy as np

def readCGNS(filename) :
    # Read CGNS file
    with h5py.File(filename, 'r') as f:

        x = f['Base/dom-1/GridCoordinates/CoordinateX/ data'][:]
        y = f['Base/dom-1/GridCoordinates/CoordinateY/ data'][:]
        z = f['Base/dom-1/GridCoordinates/CoordinateZ/ data'][:]

        VelocityX = f['Base/dom-1/FLOW_SOLUTION_CC/VelocityX/ data'][:]
        VelocityY = f['Base/dom-1/FLOW_SOLUTION_CC/VelocityY/ data'][:]
        VelocityZ = f['Base/dom-1/FLOW_SOLUTION_CC/VelocityZ/ data'][:]

        # eleme2nodes = f['Base/dom-1/Elements_NGON_n/ElementConnectivity/ data'][:]
        # elem2nodesIndex = f['Base/dom-1/Elements_NGON_n/ElementStartOffset/ data'][:]

        # X = f['WallBase/wall/WALL_FLOW_SOLUTION_CC/xWall/ data'][:]
        # Y = f['WallBase/wall/WALL_FLOW_SOLUTION_CC/yWall/ data'][:]
        # Cp = f['WallBase/wall/WALL_FLOW_SOLUTION_CC/pressureWall/ data'][:]

        # it = f['Base/GlobalConvergenceHistory/IterationCounters/ data'][:]
        # time = f['Base/GlobalConvergenceHistory/Time/ data'][:]
        # res = f['Base/GlobalConvergenceHistory/Residual/ data'][:]
        # resPhi = f['Base/GlobalConvergenceHistory/ResPhi/ data'][:]
        # cl = f['Base/GlobalConvergenceHistory/Cl/ data'][:]
        # cd = f['Base/GlobalConvergenceHistory/Cd/ data'][:]
        # cm = f['Base/GlobalConvergenceHistory/Cm/ data'][:]
        # circulation = f['Base/GlobalConvergenceHistory/Circulation/ data'][:]

        # phi = f['Base/dom-1/FLOW_SOLUTION_CC/phi/ data'][:]
        # u = f['Base/dom-1/FLOW_SOLUTION_CC/u/ data'][:]
        # v = f['Base/dom-1/FLOW_SOLUTION_CC/v/ data'][:]
        # rho = f['Base/dom-1/FLOW_SOLUTION_CC/rho/ data'][:]

        # time = np.zeros_like(it)  # Placeholder for time, as it is not available in the file
        # resPhi = np.zeros_like(it)  # Placeholder for resPhi, as it is not available in the file
        # circulation = np.zeros_like(it)  # Placeholder for circulation, as it is not available in the file

        # res = res / res[0]

        data = {}

        data['x'] = x
        data['y'] = y
        data['z'] = z
        data['VelocityX'] = VelocityX
        data['VelocityY'] = VelocityY
        data['VelocityZ'] = VelocityZ

        return data
    
errorList = []
cells = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
for N in cells :
    data = readCGNS(f'../output/convergence_green_gauss/cylinder_{N}x{N}.cgns')

    vel_mag = np.sqrt(data['VelocityX']**2 + data['VelocityY']**2 + data['VelocityZ']**2)
    # Normalized L2 (RMS) error. NOTE: using sqrt(sum()) is not grid-independent and
    # will shift the apparent convergence rate in 2D by about -1.
    error_rms = np.sqrt(np.sum((vel_mag - 1.0)**2) / vel_mag.size)
    errorList.append(error_rms)

errorList = np.array(errorList)
h = 1/np.array(cells)

plt.figure()
plt.loglog(h, errorList, 'o')
for n in cells :
    plt.text(1.1*h[cells.index(n)], 0.8*errorList[cells.index(n)], f'{n}', fontsize=12)
plt.xlabel('Mesh size h')
plt.ylabel('L2 Error in Velocity Magnitude')
plt.grid(True)

# plot convergence rate line

order = np.polyfit(np.log(h[-3:]), np.log(errorList[-3:]), 1)
rate = order[0]

plt.loglog(h, np.exp(order[1]) * h**rate, 'k--', label=f'Order ~ {rate:.4f}')
plt.legend()
plt.title('Convergence of Velocity Magnitude Error')
plt.tight_layout()
plt.savefig('../output/convergence_green_gauss/convergence_plot.pdf')