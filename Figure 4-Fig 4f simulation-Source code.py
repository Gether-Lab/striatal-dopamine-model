#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 21:03:58 2025

@author: ejdrup
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


def do_timestep_3D(u0, space_DAT, DAT_mult, Vmax, Km):
    u = u0.copy()
    
    # Propagate with forward-difference in time, central-difference in space
    u = u0 + Ds * dt * (
        (np.roll(u0, 1, axis = 0) - \
         2*u0 + \
         np.roll(u0, -1, axis = 0))  /dx_dy**2 
              + \
        (np.roll(u0, 1, axis = 1) - \
         2*u0 + \
         np.roll(u0, -1, axis = 1))  /dx_dy**2
              + \
        (np.roll(u0, 1, axis = 2) - \
          2*u0 + \
          np.roll(u0, -1, axis = 2))  /dx_dy**2)
    
    
    # Simulate reuptake
    u = u - (dt*(Vmax*DAT_mult*u)/(Km + u))*space_DAT
    
    u0 = u.copy()
    return u0, u


def diffuse_3D_range(time, space0, space_ph, space_DAT, DAT_mult, Vmax, Km, avg_release, Hz = 0.001):
    for i in tqdm(range(int(time/dt)-1)):
        #Add avg DA release to each voxel
        space_ph += (avg_release*dt)
        
        # Apply gradient operator to simulate diffusion
        space_ph, u = do_timestep_3D(space_ph, space_DAT, DAT_mult, 
                                      Vmax, Km)
        
        # Save snapshot at specified Hz
        if i%int(Hz/dt) == 0:
            space0[int(i/(Hz/dt)+1),:,:,:] = space_ph
            
        
    return space0

def create_clusters(matrix_size, num_circles, radii):
    # Initialize a list to hold the matrices for different radii
    matrices = []

    # Keep track of the centers
    centers = []

    # Randomly select centers for the circles
    for _ in range(num_circles):
        while True:
            # Randomly select the center for the circle
            max_radius = max(radii)
            center_x = np.random.randint(max_radius, matrix_size - max_radius)
            center_y = np.random.randint(max_radius, matrix_size - max_radius)

            # Check if the new circle overlaps with existing ones
            overlap = False
            for cx, cy in centers:
                if (center_x - cx)**2 + (center_y - cy)**2 < (2 * max_radius)**2:
                    overlap = True
                    break
            
            # If no overlap, break the loop and place the circle
            if not overlap:
                centers.append((center_x, center_y))
                break

    # Create a separate matrix for each radius
    for radius in radii:
        # Initialize a 2D matrix filled with zeros
        matrix = np.zeros((matrix_size, matrix_size))
        
        # Create circles at the same centers with the current radius
        for center_x, center_y in centers:
            for x in range(max(0, center_x - radius), min(matrix_size, center_x + radius)):
                for y in range(max(0, center_y - radius), min(matrix_size, center_y + radius)):
                    # Check if the current position is within the current radius
                    if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                        matrix[x, y] = 1
        
        # # Adjust the drawing logic for radius 1 to create a 2x2 square
        if radius == 1:
            for center_x, center_y in centers:
                matrix[center_x:center_x + 4, center_y:center_y + 4] = 1
        else:
            # Create circles at the same centers with the current radius
            for center_x, center_y in centers:
                for x in range(max(0, center_x - radius), min(matrix_size, center_x + radius)):
                    for y in range(max(0, center_y - radius), min(matrix_size, center_y + radius)):
                        # Check if the current position is within the current radius
                        if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                            matrix[x, y] = 1

        # Add the matrix for the current radius to the list
        matrices.append(matrix)

    return matrices
#%%

# Moles of DA released per um^3 with a single vesicle
Q_eff = 2.3374945339209775e-05
Active_terminal_density = 0.04
f_rate = 4
pr_release = 0.06

#DA released per voxel on avg (in nM)
avg_release = Q_eff*f_rate*pr_release*Active_terminal_density

#%% Define parameters

# Start concentration
start_conc = 15*10**-9 # in molar

# Time parameters
time = 0.5 # Duration of simulation in sec
Hz = 0.001 # Sampling rate in sec

# Diffusivity of DA in striatum
D = 763 # um2.s-1
gamma = 1.54
Ds = D/(gamma**2)

# simulation size, um
w = h = 1.8
depth = 3.52 * 2

# intervals in x-, y- directions, um
dx = dy = dz = 0.02 #0.02
dx_dy = dx
nx, ny, nz = int(w/dx), int(h/dy), int(depth/dz),

# Calculate timestep
dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

# Cluster parameters
num_clusters = 8   # Define the number of clusters
radii = [0.02]  # List of different radii to use
radii_conv = [round(x/dx) for x in radii] # convert to size for simulations

# Simulate DAT clusters
DAT_clusters = create_clusters(int(h/dx), num_clusters, radii_conv)

# Add uniform at the end

# DAT_clusters.append(np.ones((90,90)))


#%% Simulation

sim_result_list = []

for i in range(len(DAT_clusters)):
    
    # Create simulation space and supporting matrixes

    # Initiate the space
    space0 = np.zeros((round(time/Hz)+1, nx, ny, nz))

    # Initiate placeholder for the loop
    space_ph = np.zeros((nx, ny, nz))
    
    # set middle voxel to start_conc and correct for ECF
    space_ph[:] = start_conc

    # Project DAT clusters to the simulation space dimensions
    space_DAT = np.zeros((nx, ny, nz))
    space_DAT[:,:,-1] = DAT_clusters[i]

    # Calculate multiplier to convert avg. uptake capacity to cluster capactity density
    # based on cluster sizes (and therefore density of DAT molecules in the cluster)
    DAT_mult = (nx*ny*nz)/np.sum(space_DAT)
    
    sim_result = diffuse_3D_range(time, space0, space_ph, space_DAT, DAT_mult, Vmax = 4*10**-6, Km = 210*10**-9, 
                                  avg_release = avg_release, Hz = Hz)
    
    sim_result_list.append(sim_result)
    

