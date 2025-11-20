#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 08:14:58 2025

@author: ejdrup
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from tqdm import tqdm
from scipy.optimize import curve_fit


def impulse_2(t,k1,k2,tau,ts):
    return np.exp(-(t+1)*(k1*tau+k2*ts))

def exp_decay(t,N0,time_constant):
    return N0*np.exp(-time_constant*t)

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

#%% Simulate percentiles across different Vmax'es
vmax_range = np.linspace(0.5*10**-6,10*10**-6,39)
vmax_percentiles_DS = np.zeros((3,len(vmax_range)))
vmax_percentiles_VS = np.zeros((3,len(vmax_range)))

simulation_DS, space_init_DS, firing_DS, release_sites_DS, var_list_DS = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)
        
simulation_VS, space_init_VS, firing_VS, release_sites_VS, var_list_VS = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25*(1/0.9), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

for i, vmax_val in enumerate(vmax_range):
    print(i)
    
    # DS
    full_sim = sim_dynamics_3D(simulation_DS, space_init_DS, release_sites_DS, firing_DS, var_list_DS, 
                     Q = 3000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_DS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
    
    
    # VS
    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 3000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
#%% Plot vmax effect on percentiles
fig, (ax1) = plt.subplots(1,1,figsize = (2.5,2.5), dpi = 400, gridspec_kw={"width_ratios": [1]})

ax1.set_title("Effect of DAT V$_{max}$ on [DA]", fontsize = 10)
ax1.set_ylabel("[DA] (nM)")
ax1.set_xlabel("V$_{max}$ (\u00B5M s$^{-1}$)")
ax1.set_ylim(0,150)
ax1.set_xlim(0*10**-6,9*10**-6)
ax1.set_xticks([0*10**-6,3*10**-6,6*10**-6,9*10**-6])
ax1.set_xticklabels([0,3,6,9])
color_list = ["black","grey","lightgrey"][::-1]

DS_max_idx = np.argmin(abs(vmax_range-(6*1.5)*10**-6))
DS_min_idx = np.argmin(abs(vmax_range-(6*0.5)*10**-6))

VS_max_idx = np.argmin(abs(vmax_range-(2*1.5)*10**-6))
VS_min_idx = np.argmin(abs(vmax_range-(2*0.5)*10**-6))

linestyles = [":","--","-"]

ax1.plot(vmax_range,vmax_percentiles_DS[1,:]*10**9, color = "cornflowerblue")
ax1.plot(vmax_range,vmax_percentiles_VS[1,:]*10**9, color = "indianred")
    # ax1.plot(vmax_range,vmax_percentiles_VS[i,:]*10**9, color = "indianred")
    
# legend = ax1.legend(("10$^{th}$","30$^{th}$","50$^{th}$","70$^{th}$","90$^{th}$"), frameon = False,
#            handlelength = 1.2, prop={'size': 9})
legend = ax1.legend(("DS", "VS"), frameon = True,
            handlelength = 1.2, prop={'size': 8}, loc = "upper right", bbox_to_anchor = [1.05,0.45])
legend.set_title('Tonic$^{50th}$',prop={'size': 8})

ax1.text(6*10**-6,55,
          "Median DS V$_{max}$\n\u00B1 50 %", ha = "center", fontsize = 8, rotation = 90)
ax1.fill_between(x = [3*10**-6,9*10**-6], y1 = [150,150], 
                  color = "cornflowerblue", alpha = 0.5, lw = 0)
ax1.text(2.1*10**-6,55,
          "Median VS V$_{max}$\n\u00B1 50 %", ha = "center", fontsize = 8, rotation = 90)
ax1.fill_between(x = [1*10**-6,3*10**-6], y1 = [150,150], 
                  color = "indianred", alpha = 0.5, lw = 0)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


fig.tight_layout()

#%% Vmax at phasic and tonic
fig, (ax1) = plt.subplots(1,1,figsize = (2.5,2.5), dpi = 400, gridspec_kw={"width_ratios": [1]})

ax1.set_title("Effect of DAT V$_{max}$ on [DA]", fontsize = 10)
ax1.set_ylabel("Relative [DA] change\n from 2 \u00B5M s$^{-1}$")
ax1.set_xlabel("V$_{max}$ (\u00B5M s$^{-1}$)")
ax1.set_ylim(0,6)
ax1.set_xlim(0*10**-6,9*10**-6)
ax1.set_xticks([0*10**-6,3*10**-6,6*10**-6,9*10**-6])
ax1.set_xticklabels([0,3,6,9])
color_list = ["black","grey","lightgrey"][::-1]

DS_max_idx = np.argmin(abs(vmax_range-(6*1.5)*10**-6))
DS_min_idx = np.argmin(abs(vmax_range-(6*0.5)*10**-6))

VS_max_idx = np.argmin(abs(vmax_range-(2*1.5)*10**-6))
VS_min_idx = np.argmin(abs(vmax_range-(2*0.5)*10**-6))

linestyles = [":","--","-"]

ax1.plot(vmax_range,vmax_percentiles_VS[2,:]/vmax_percentiles_VS[2,6], color = "indianred", ls = "-")
ax1.plot(vmax_range,vmax_percentiles_VS[1,:]/vmax_percentiles_VS[1,6], color = "indianred", ls = ":")

ax1.hlines(1,0,9*10**-6, color = "k", lw = 0.8, ls = ":")

# ax1.plot(vmax_range,vmax_percentiles_VS[2,:]/vmax_percentiles_VS[2,22], color = "cornflowerblue", ls = "-")
# ax1.plot(vmax_range,vmax_percentiles_VS[1,:]/vmax_percentiles_VS[1,22], color = "cornflowerblue", ls = ":")

    # ax1.plot(vmax_range,vmax_percentiles_VS[i,:]*10**9, color = "indianred")
    
# legend = ax1.legend(("10$^{th}$","30$^{th}$","50$^{th}$","70$^{th}$","90$^{th}$"), frameon = False,
#            handlelength = 1.2, prop={'size': 9})
legend = ax1.legend(("Peak$^{99.5th}$", "Tonic$^{50th}$"), frameon = False,
            handlelength = 1.2, prop={'size': 8}, loc = "upper right", bbox_to_anchor = [1.05,0.95])
# legend.set_title('Tonic$^{50th}$',prop={'size': 8})

# ax1.text(6*10**-6,55,
#           "Median DS V$_{max}$\n\u00B1 50 %", ha = "center", fontsize = 8, rotation = 90)
# ax1.fill_between(x = [3*10**-6,9*10**-6], y1 = [150,150], 
#                   color = "cornflowerblue", alpha = 0.5, lw = 0)
ax1.text(2.1*10**-6,2.3,
          "Median VS V$_{max}$\n\u00B1 50 %", ha = "center", fontsize = 8, rotation = 90)
ax1.fill_between(x = [1*10**-6,3*10**-6], y1 = [150,150], 
                  color = "indianred", alpha = 0.5, lw = 0)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)



# ax2.set_title("DAT effect V$_{max}$ on [DA]", fontsize = 10)
# ax2.set_xlabel("DA Percentiles")
# ax2.set_ylim(0,150)
# ax2.set_ylabel("nM")

# for i in range(3):
#     ax2.plot([i-0.1,i-0.1],[vmax_percentiles_DS[i,DS_max_idx]*10**9,vmax_percentiles_DS[i,DS_min_idx]*10**9],
#              color = "cornflowerblue", lw = 2)
#     ax2.plot([i+0.1,i+0.1],[vmax_percentiles_VS[i,VS_max_idx]*10**9,vmax_percentiles_VS[i,VS_min_idx]*10**9],
#              color = "indianred", lw = 2)
    
# ax2.set_xticks([0,1,2])
# ax2.set_xticklabels(["10$^{th}$","50$^{th}$","90$^{th}$"])
# ax2.spines["top"].set_visible(False)
# ax2.spines["right"].set_visible(False)

fig.tight_layout()