#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:39:58 2025

@author: ejdrup
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from tqdm import tqdm
from scipy.optimize import curve_fit


def sim_space_neurons_3D(width = 100, depth = 10, dx_dy = 1, time = 1, D = 763, 
              inter_var_distance = 2.92, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01):
    ## Generate overall simulation space
    
    # Simulation time, sec.
    t = time
    # Field size, um
    w = h = width
    # Depth of field
    depth = depth
    # Intervals in x-, y- directions, um
    dx = dy = dz = dx_dy
    # Steps per side
    nx, ny, nz = int(w/dx), int(h/dy), int(depth/dz)
    # Calculate time step
    dx2, dy2 = dx*dx, dy*dy
    dt = dx2 * dy2 / (2 * D * (dx2 + dy2))
    # Generate simulation space (one snapshot every 10 ms)
    space0 = np.zeros((int(t/Hz), nx, ny, nz))
    # Prep for image of all release sites
    space_ph = np.zeros((nx, ny, nz))
    
    ## Generate firing pattern
    p_r = p_r
    # Number of varicosities
    n_varico = int((width**2*depth)/inter_var_distance)
    # Generate varicosities from random linear distribution
    x_varico = np.random.randint(0, high = (w/dx), size = n_varico)
    y_varico = np.random.randint(0, high = (w/dy), size = n_varico)
    z_varico = np.random.randint(0, high = (depth/dz), size = n_varico)
    
    # Assign neuron identity to terminals
    neuro_identity = np.random.randint(0, high = n_neurons, size = n_varico)
    # Firing pattern of each neuron
    neuron_firing = np.random.poisson(f_rate*dt,(n_neurons,int(t/dt)))
    neuron_firing[neuron_firing > 1] = 1 # avoid multiple release events on top of each other
    
    # Firing pattern of each terminal
    firing = neuron_firing[neuro_identity,:]
    # Add indv release prob of 0.06
    terminal_events = np.where(firing)
    terminal_refraction = np.random.choice(len(terminal_events[0]),
                     int(len(terminal_events[0])*(1-p_r)),
                     replace = False)

    firing[terminal_events[0][terminal_refraction],terminal_events[1][terminal_refraction]] = 0
    # terminal_refraction = np.random.poisson(1-p_r,firing.shape)
    # terminal_refraction[terminal_refraction > 1] = 1 # avoid multiple release events on top of each other
    # terminal_refraction = (terminal_refraction-1)*-1 # flip logic

    return space0, space_ph, firing.T, np.array([x_varico,y_varico, z_varico]), np.array([time, dt, dx_dy, inter_var_distance, Hz])

def sim_dynamics_3D(space0, space_ph, release_sites, firing, var_list, 
                 Q = 3000, uptake_rate = 4*10**-6, Km = 210*10**-9,
                 Ds = 321.7237308146399, ECF = 0.21):
    # print(uptake_rate)
    # print(Q)
    # Extract parameters
    t = var_list[0]
    dt = var_list[1]
    dx_dy = var_list[2]
    Hz = var_list[4]
    
    # DA release per vesicle
    single_vesicle_vol = (4/3*np.pi*(0.025)**3) 
    voxel_volume = (dx_dy)**3 # Volume of single voxel
    single_vesicle_DA = 0.025 * Q/1000 # 0.025 M at Q = 1000
    Q_eff = single_vesicle_vol/voxel_volume * single_vesicle_DA * 1/ECF
    
    
    for i in tqdm(range(int(t/dt)-1)):
        
        # Add release events per time step
        space_ph[release_sites[0,:][np.where(firing[i,:])],
               release_sites[1,:][np.where(firing[i,:])],
               release_sites[2,:][np.where(firing[i,:])]] += Q_eff
        
        # Apply gradient operator to simulate diffusion
        space_ph, u = do_timestep_3D(space_ph, 
                                      uptake_rate, Ds, dt, dx_dy, Km)
        
        # Save snapshot at specified Hz
        if i%int(Hz/dt) == 0:
            space0[int(i/(Hz/dt)),:,:,:] = space_ph
            
        
    return space0

def do_timestep_3D(u0, uptake_rate, Ds, dt, dx_dy, Km):
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
    u = u - dt*(uptake_rate*u)/(Km + u)
    
    u0 = u.copy()
    return u0, u

#%% Size difference

# Small
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 10, depth = 10, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.005)
     
full_sim_small = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 6*10**-6, Ds = 321.7237308146399)


# Medium sized
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.005)
     
full_sim_medium = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 6*10**-6, Ds = 321.7237308146399)


# Large
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.005)
     
full_sim_large = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 6*10**-6, Ds = 321.7237308146399)


#%% Extract percentiles and histogram

dist_DA_small = np.percentile(full_sim_small[100:,:,:,:],np.linspace(0,100,1001))
dist_DA_medium = np.percentile(full_sim_medium[100:,:,:,:],np.linspace(0,100,1001))
dist_DA_large = np.percentile(full_sim_large[100:,:,:,:],np.linspace(0,100,1001))

#%% Grid figure
fig, (ax1, ax2) = plt.subplots(2,1,figsize = (1.5,2.5), dpi = 400, gridspec_kw={"height_ratios": [1,1]})

ax1.set_title("Diameter", fontsize = 10)
ax1.set_ylabel("50 \u00B5m", fontsize = 10)
ax1.imshow(np.log10(full_sim_large[154,10,25:-25,25:-25]), vmax = -6.3, vmin = -8.5)
ax1.set_ylim(-25,75)
ax1.set_xlim(-25,75)
ax1.set_xticks([])
ax1.set_yticks([])
# ax1.axis(False)

ax2.set_ylabel("100 \u00B5m", fontsize = 10)
# ax2.imshow(full_sim_coarse[200,10,:,:])
ax2.imshow(np.log10(full_sim_large[154,10,:,:]), vmax = -6.3, vmin = -8.5)
ax2.set_ylim(0,99)
ax2.set_xlim(0,99)
ax2.set_xticks([])
ax2.set_yticks([])
# ax2.axis(False)

fig.tight_layout()
#%% Quantification difference

fig, (ax1) = plt.subplots(1,1,figsize = (3,2.5), dpi = 400, gridspec_kw={"width_ratios": [1]})

x_range = np.linspace(0,100,1001)
ax1.set_title("Effect of simulation size", fontsize = 10)
ax1.set_ylabel("[DA] (nM)")
ax1.set_xlabel("Percentile")
ax1.plot(x_range,dist_DA_small*10**9, "palevioletred")
ax1.plot(x_range,dist_DA_medium*10**9, "darkkhaki")
ax1.plot(x_range,dist_DA_large*10**9, "seagreen", ls = "--")
# ax1.legend(("0.2 \u00B5m","1.0 \u00B5m","Difference"), 
#            frameon = False, loc = "lower center", ncol = 3, handlelength = 1.2, handletextpad = 0.5,
#            columnspacing = 0.5,fontsize = 8)
ax1.legend(("10 \u00B5m","50 \u00B5m","100 \u00B5m"), title = "Diameter     ", title_fontsize = 9,
           frameon = False, loc = "upper left", ncol = 1, handlelength = 1.2, handletextpad = 0.8,
           columnspacing = 0.5,fontsize = 9, bbox_to_anchor = [0,1.05])

ax1.set_yscale("log")
ax1.set_ylim(0.5,100)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()


#%% Grainedness
# Really fine grained
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 20, depth = 20, dx_dy = 0.1, time = 4, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.005)
     
full_sim_fine = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 2*10**-6, Ds = 321.7237308146399)

# Fine grained
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 20, depth = 20, dx_dy = 0.5, time = 4, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.005)
     
full_sim_medium = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 2*10**-6, Ds = 321.7237308146399)


# Coarse grained
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 20, depth = 20, dx_dy = 1, time = 4, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.005)
     
full_sim_coarse = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 2*10**-6, Ds = 321.7237308146399)

# CReally coarse grained
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 20, depth = 20, dx_dy = 2, time = 4, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.005)
     
full_sim_coarse2x = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 2*10**-6, Ds = 321.7237308146399)


 #%% Extract percentiles and histogram

dist_DA_fine = np.percentile(full_sim_fine[50:,:,:,:],np.linspace(0,100,4001))
dist_DA_medium = np.percentile(full_sim_medium[50:,:,:,:],np.linspace(0,100,4001))
dist_DA_coarse = np.percentile(full_sim_coarse[50:,:,:,:],np.linspace(0,100,4001))
dist_DA_coarse2x = np.percentile(full_sim_coarse2x[50:,:,:,:],np.linspace(0,100,4001))

# hist_DA_fine, _ = np.histogram(np.log10(full_sim_fine[50:,:,:,:]).flatten(), bins = np.log10(np.logspace(-10,-5,1000)), density = True)
# hist_DA_coarse, _ = np.histogram(np.log10(full_sim_coarse[50:,:,:,:]).flatten(), bins = np.log10(np.logspace(-10,-5,1000)), density = True)

#%% Grid figure
fig, (ax1, ax2) = plt.subplots(2,1,figsize = (1.5,2.5), dpi = 400, gridspec_kw={"height_ratios": [1,1]})

ax1.set_title("Granularity", fontsize = 10)
ax1.set_ylabel("0.2 \u00B5m", fontsize = 10)
ax1.imshow(full_sim_fine[281,10,:,:])
ax1.set_xticks([])
ax1.set_yticks([])
# ax1.axis(False)

ax2.set_ylabel("1.0 \u00B5m", fontsize = 10)
# ax2.imshow(full_sim_coarse[200,10,:,:])
ax2.imshow(full_sim_fine[281,10,::5,::5])
ax2.set_xticks([])
ax2.set_yticks([])
# ax2.axis(False)

fig.tight_layout()
#%% Quantification difference

fig, (ax1) = plt.subplots(1,1,figsize = (3,2.5), dpi = 400, gridspec_kw={"width_ratios": [1]})

x_range = np.linspace(0,100,4001)
ax1.set_title("Effect of simulation granularity", fontsize = 10)
ax1.set_ylabel("[DA] (nM)")
ax1.set_xlabel("Percentile")
ax1.plot(x_range,dist_DA_fine*10**9, "palevioletred", ls = "-")
ax1.plot(x_range,dist_DA_medium*10**9, "darkkhaki", ls = ":")
ax1.plot(x_range,dist_DA_coarse*10**9, "darkgreen", ls = ":")
ax1.plot(x_range,dist_DA_coarse2x*10**9, "dimgrey", ls = ":")

# ax1.plot(x_range,abs(dist_DA_fine-dist_DA_medium)*10**9, color = "dimgrey", ls ="-")
# ax1.plot(x_range,abs(dist_DA_fine-dist_DA_coarse)*10**9, color = "dimgrey", ls = "--")
# ax1.plot(x_range,abs(dist_DA_fine-dist_DA_coarse2x)*10**9, color = "dimgrey", ls = ":")
ax1.legend(("0.1 \u00B5m","0.5 \u00B5m","1.0 \u00B5m","2.0 \u00B5m"), 
            frameon = False, loc = "upper left", ncol = 1, handlelength = 1.2, handletextpad = 0.5,
            columnspacing = 0.5,fontsize = 8)
# ax1.legend(("0.2 \u00B5m","1.0 \u00B5m","Difference"), 
#            frameon = False, loc = "upper left", ncol = 1, handlelength = 1.2, handletextpad = 0.8,
#            columnspacing = 0.5,fontsize = 9)

ax1.set_yscale("log")
ax1.set_ylim(10,10000)
ax1.set_xlim(0,100)

axins = ax1.inset_axes([0.5, 0.34, 0.3, 0.66])
axins.plot(x_range,dist_DA_fine*10**9, "palevioletred", ls = "-")
axins.plot(x_range,dist_DA_medium*10**9, "darkkhaki", ls = "--")
axins.plot(x_range,dist_DA_coarse*10**9, "darkgreen", ls = ":")
axins.plot(x_range,dist_DA_coarse2x*10**9, "k", ls = ":")
#axins.legend(("Saline", "Amph."), loc = "upper right", frameon = False, handlelength = 1.5)
# axins.fill_between(np.linspace(-20,100,no_blocks),mean_amph-sem_amph,mean_amph+sem_amph, color = "mediumpurple", alpha = 0.3, lw = 0)
# axins.fill_between(np.linspace(-20,100,no_blocks),mean_sal-sem_sal,mean_sal+sem_sal, color = "dimgrey", alpha = 0.3, lw = 0)
#axins.vlines(0,-2,2,lw = 0.8, ls = '--', color = "black")
axins.tick_params(axis = "x", length = 3)
axins.tick_params(axis = "y", length = 0)
# axins.set_xscale("log")
x1, x2, y1, y2 = 99.75, 100, 100, 10000
axins.set_yscale("log")
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels(["99.75", "100"])
axins.set_yticklabels('')
axins.spines['top'].set_color('k')
axins.spines['left'].set_color('k')
axins.spines['right'].set_color('k')
axins.spines['bottom'].set_color('k')
axins.set_alpha(0.5)
ax1.indicate_inset_zoom(axins, edgecolor="grey", alpha = 0.5, clip_on = False)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()

#%% Quantification difference - zoom high conc

fig, (ax1) = plt.subplots(1,1,figsize = (3,2.5), dpi = 400, gridspec_kw={"width_ratios": [1]})

x_range = np.linspace(0,100,4001)
ax1.set_title("Difference from 0.1 \u00B5m granularity", fontsize = 10)
ax1.set_ylabel("[DA] (nM)")
ax1.set_xlabel("Percentile")
# ax1.plot(x_range,dist_DA_fine*10**9, "palevioletred", ls = "-")
# ax1.plot(x_range,dist_DA_medium*10**9, "darkkhaki", ls = "--")
# ax1.plot(x_range,dist_DA_coarse*10**9, "palevioletred", ls = ":")
# ax1.plot(x_range,dist_DA_coarse2x*10**9, "blue", ls = ":")

ax1.plot(x_range,abs(dist_DA_fine-dist_DA_medium)*10**9, color = "darkkhaki", ls ="-")
ax1.plot(x_range,abs(dist_DA_fine-dist_DA_coarse)*10**9, color = "darkgreen", ls = "-")
ax1.plot(x_range,abs(dist_DA_fine-dist_DA_coarse2x)*10**9, color = "k", ls = "-")
ax1.legend(("0.5 \u00B5m","1.0 \u00B5m","2.0 \u00B5m"), 
            frameon = False, loc = "upper left", ncol = 1, handlelength = 1.2, handletextpad = 0.5,
            columnspacing = 0.5, title = "Voxel diameter", title_fontsize = 8, fontsize = 8)
# ax1.legend(("0.2 \u00B5m","1.0 \u00B5m","Difference"), 
#            frameon = False, loc = "upper left", ncol = 1, handlelength = 1.2, handletextpad = 0.8,
#            columnspacing = 0.5,fontsize = 9)

ax1.set_yscale("log")
ax1.set_ylim(0.01,1000)
ax1.set_xlim(0,100)

axins = ax1.inset_axes([0.6, 0.42, 0.3, 0.58])

axins.plot(x_range,abs(dist_DA_fine-dist_DA_medium)*10**9, color = "darkkhaki", ls ="-")
axins.plot(x_range,abs(dist_DA_fine-dist_DA_coarse)*10**9, color = "darkgreen", ls = "-")
axins.plot(x_range,abs(dist_DA_fine-dist_DA_coarse2x)*10**9, color = "k", ls = "-")
#axins.legend(("Saline", "Amph."), loc = "upper right", frameon = False, handlelength = 1.5)
# axins.fill_between(np.linspace(-20,100,no_blocks),mean_amph-sem_amph,mean_amph+sem_amph, color = "mediumpurple", alpha = 0.3, lw = 0)
# axins.fill_between(np.linspace(-20,100,no_blocks),mean_sal-sem_sal,mean_sal+sem_sal, color = "dimgrey", alpha = 0.3, lw = 0)
#axins.vlines(0,-2,2,lw = 0.8, ls = '--', color = "black")
axins.tick_params(axis = "x", length = 3)
axins.tick_params(axis = "y", length = 0)
# axins.set_xscale("log")
x1, x2, y1, y2 = 99, 100, 1, 1000
axins.set_yscale("log")
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels(["99", "100"])
axins.set_yticklabels('')
axins.spines['top'].set_color('k')
axins.spines['left'].set_color('k')
axins.spines['right'].set_color('k')
axins.spines['bottom'].set_color('k')
axins.set_alpha(0.5)
ax1.indicate_inset_zoom(axins, edgecolor="grey", alpha = 0.5, clip_on = False)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()