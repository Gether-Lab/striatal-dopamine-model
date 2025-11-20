#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:58:39 2025

@author: ejdrup
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


def do_timestep_3D(u0, uptake_rate):
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
    # u = u - dt*(uptake_rate*u)/(Km + u)
    
    u0 = u.copy()
    return u0, u


def diffuse_3D_range(ms, space0, vmax):
    for i in tqdm(range(int(time/dt)-1)):
        
        _, space0[i+1,:,:,:] = do_timestep_3D(space0[i,:,:,:], 
                                      vmax)
    return space0

def point_source_3D(UCf,Ds,t,r,vmax,km):
    
    C = (UCf/(0.21*(4*Ds*t*np.pi)**(3/2)))*np.exp(-r**2/(4*Ds*t))*np.exp(-(vmax/km)*t)
    
    return C

#%% Size definitions 
time = 0.1 # in sec

# field size, um
w = h = depth = 20
# intervals in x-, y- directions, um
dx = dy = dz = 0.25
dx_dy = dx
# Diffusivity of DA in striatum, um2.s-1
D = 763
gamma = 1.54
Ds = D/(gamma**2)

nx, ny, nz = int(w/dx), int(h/dy), int(depth/dz),

dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

space0 = np.zeros((int(time/dt), nx, ny, nz))

# Q of 3000 to a concentration in dx**3 volume
Q = 3000
area = (dx*10**-6)**3 # in m^3
area_L = area*1000 # in liters
Na = 6.022*10**23 # Avrogadros number
start_conc = Q/Na/area_L*10**9 # start conc of single voxel in nM

# set middle voxel to start_conc and correct for ECF
space0[0, int(w/2/dx),int(h/2/dy), int(depth/2/dz)] = start_conc*(1/0.21) 

# Q of 3000 to a concentration in a single point
U = (4/3*np.pi*(25*10**-3)**3)*1000 # Volume in uL
Cf = 0.025375*10**6 # Fill concentration in uM at Q = 1000
Q_factor = 3 # Adjust to Q = 3000
UCf = U*Cf*Q_factor

#%% Run diffusion simulation and analytical result

sim_result = diffuse_3D_range(int(time*1000),space0, 0)

#%%
radius_range = np.linspace(dx,w/2,int(w/2/dx))
time_range = np.linspace(dt,time,int(time/dt))

analytical_result_t1 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = 0.005, 
                            r = radius_range,
                            vmax = 0, 
                            km = 0.210)

analytical_result_t2 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = 0.01, 
                            r = radius_range,
                            vmax = 0, 
                            km = 0.210)

analytical_result_t3 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = 0.02, 
                            r = radius_range,
                            vmax = 0, 
                            km = 0.210)

analytical_result_r2 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = time_range, 
                            r = 2,
                            vmax = 0, 
                            km = 0.210)

analytical_result_r3 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = time_range, 
                            r = 3,
                            vmax = 0, 
                            km = 0.210)

analytical_result_r5 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = time_range, 
                            r = 5,
                            vmax = 0, 
                            km = 0.210)


analytical_result_t4 = point_source_3D(UCf = UCf, 
                            Ds = D/(gamma**2),
                            t = 0.002, 
                            r = radius_range,
                            vmax = 0, 
                            km = 0.210)
#%% Plot across radius single plot

# fig, (ax1, ax2) = plt.subplots(1,2,figsize = (3.2,3), dpi = 400)
fig = plt.figure(figsize = (3.5,2.5), dpi = 400)
gs = GridSpec(2,2, height_ratios=[1,1], width_ratios=[1.5,0.8])

ax11 = fig.add_subplot(gs[0,0])
ax12 = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[:,1])

ax11.set_title("5 ms", fontsize = 10)
im = ax11.imshow(sim_result[int(0.005/dt),:,:,40], vmin = 0, vmax = 200, cmap = "magma")
ax11.set_xlim(0,80)
ax11.set_ylim(0,80)
ax11.set_xticks([])
ax11.set_yticks([])
# ax11.set_xlabel("10 \u00B5m")
ax11.set_ylabel("10 \u00B5m")
ax11.spines["left"].set_visible(False)
ax11.spines["bottom"].set_visible(False)
ax11.spines["right"].set_visible(False)
ax11.spines["top"].set_visible(False)

ax12.set_title("10 ms", fontsize = 10)
ax12.imshow(sim_result[int(0.01/dt),:,:,40], vmin = 0, vmax = 200, cmap = "magma")
ax12.set_xlim(0,80)
ax12.set_ylim(0,80)
ax12.set_xticks([])
ax12.set_yticks([])
ax12.set_xlabel("10 \u00B5m")
ax12.set_ylabel("10 \u00B5m")
ax12.spines["left"].set_visible(False)
ax12.spines["bottom"].set_visible(False)
ax12.spines["right"].set_visible(False)
ax12.spines["top"].set_visible(False)



ax2.set_title("Comparison", fontsize = 10)

# ax2.plot([],[], color = "dimgrey", ls = "-")
# ax2.plot([],[], color = "dimgrey", ls = "--")
# ax2.plot([],[], color = "dimgrey", ls = ":")
# ax2.plot([],[], color = "black", ls = "-")
# ax2.plot([],[], color = "green", ls = "-")




# Analytical plot
ax2.plot(radius_range,analytical_result_t4,
          color = "black", ls = "-")
ax2.plot(radius_range,analytical_result_t1,
          color = "black", ls = "--")
ax2.plot(radius_range,analytical_result_t2,
         color = "black", ls = "-.")
ax2.plot(radius_range,analytical_result_t3,
          color = "black", ls = ":")

ax2.set_ylabel("[DA] (nM)", labelpad = 8)
ax2.set_ylim(0,300)
# ax2.set_yticks([0,200,400,600,800,1000])

legend1 = plt.legend(("2 ms","5 ms", "10 ms", "20 ms"), frameon = False,
            loc = "upper right", handlelength = 1.4, bbox_to_anchor = [1.1,1.05], fontsize = 7)
legend1.set_title('Analytical',prop={'size':7})
plt.gca().add_artist(legend1)


# Simulation plot
sim1, = ax2.plot(radius_range,sim_result[int(0.002/dt),40:,40,40],
         color = "darkgreen", ls = "-")
sim2, = ax2.plot(radius_range,sim_result[int(0.005/dt),40:,40,40],
         color = "darkgreen", ls = "--")
sim3, = ax2.plot(radius_range,sim_result[int(0.01/dt),40:,40,40],
          color = "darkgreen", ls = "-.")
sim4, = ax2.plot(radius_range,sim_result[int(0.02/dt),40:,40,40],
          color = "darkgreen", ls = ":")


legend = ax2.legend(handles = [sim1, sim2, sim3, sim4], labels = ["2 ms","5 ms", "10 ms", "20 ms"], frameon = False,
            loc = "upper right", handlelength = 1.4, bbox_to_anchor = [1.1,0.57], fontsize = 7)
legend.set_title('Simulation',prop={'size':7})




ax2.set_xlabel("radius (\u00B5m)")
ax2.set_xlim(0,10)
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)

fig.tight_layout(w_pad = -1)

cbar_ax = fig.add_axes([0.02, 0.265, 0.015, 0.555])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_yticks([])
# cbar_ax.set_xlim(0,1)
cbar_ax.set_title('[DA]', fontsize = 8)