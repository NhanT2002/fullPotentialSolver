#!/usr/bin/env python3
"""
Plot Cl-alpha hysteresis from unsteady simulation output
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

# Read time history from CGNS file
cgns_file = '../output/output_168.cgns'

with h5py.File(cgns_file, 'r') as f:
    # Read from GlobalConvergenceHistory
    gch = f['Base/GlobalConvergenceHistory']
    
    time = gch['Time/ data'][:]
    alpha = gch['Alpha/ data'][:]
    cl = gch['Cl/ data'][:]
    cd = gch['Cd/ data'][:]
    
    # Filter out zero entries (if array was pre-allocated)
    valid = time > 0
    if time[0] == 0:
        valid[0] = True  # Keep t=0
    
    time = time[valid]
    alpha = alpha[valid]
    cl = cl[valid]
    cd = cd[valid]
    
    print(f"Loaded {len(time)} time steps")
    print(f"Time range: [{time.min():.1f}, {time.max():.1f}]")
    print(f"Alpha range: [{alpha.min():.2f}, {alpha.max():.2f}] deg")
    print(f"Cl range: [{cl.min():.4f}, {cl.max():.4f}]")

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Cl-alpha hysteresis
ax1 = axes[0]
ax1.plot(alpha, cl, 'b-', linewidth=1.5, label='Unsteady')
ax1.plot(alpha[0], cl[0], 'go', markersize=10, label='Start')
ax1.plot(alpha[-1], cl[-1], 'r^', markersize=10, label='End')
ax1.set_xlabel(r'$\alpha$ (deg)', fontsize=12)
ax1.set_ylabel(r'$C_l$', fontsize=12)
ax1.set_title('Cl-Alpha Hysteresis Loop')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add arrows to show direction
n = len(alpha)
for i in [n//6, n//3, n//2, 2*n//3, 5*n//6]:
    if i < n-1:
        ax1.annotate('', xy=(alpha[i+1], cl[i+1]), xytext=(alpha[i], cl[i]),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

# Plot 2: Time history
ax2 = axes[1]
ax2.plot(time, alpha, 'r-', label=r'$\alpha$ (deg)')
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel(r'$\alpha$ (deg)', color='r', fontsize=12)
ax2.tick_params(axis='y', labelcolor='r')

ax2b = ax2.twinx()
ax2b.plot(time, cl, 'b-', label=r'$C_l$')
ax2b.set_ylabel(r'$C_l$', color='b', fontsize=12)
ax2b.tick_params(axis='y', labelcolor='b')

ax2.set_title('Time History')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hysteresis_plot.png', dpi=150)
print("Saved: hysteresis_plot.png")
plt.show()
