#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:39:36 2025

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

#%% Km testing

km_range = np.linspace(210,6000,25)*10**-9
# km_range = np.linspace(1*10**-9,50000*10**-9,20)
km_percentiles_DS = np.zeros((3,len(km_range)))
km_mean_DS = np.zeros((len(km_range),))
km_percentiles_VS = np.zeros((3,len(km_range)))
km_mean_VS = np.zeros((len(km_range),))

# DS
simulation, space_init, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 25, depth = 25, dx_dy = 1, time = 30, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

for i, km_val in enumerate(km_range):
    print(i)
    full_sim = sim_dynamics_3D(simulation, space_init, release_sites, firing, var_list, 
                     Q = 3000, uptake_rate = 4.5*10**-6, Km = km_val, Ds = 321.7237308146399)
    
    km_percentiles_DS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
    km_mean_DS[i] = np.mean(full_sim[int(full_sim.shape[0]/2):,:,:,:])
 
# VS    
simulation, space_init, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 25, depth = 25, dx_dy = 1, time = 30, D = 763,
                  inter_var_distance = (1/0.85)*25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)
        
for i, km_val in enumerate(km_range):
    print(i)
    full_sim = sim_dynamics_3D(simulation , space_init, release_sites, firing, var_list, 
                     Q = 3000, uptake_rate = 1.55*10**-6, Km = km_val, Ds = 321.7237308146399)
    
    km_percentiles_VS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
    km_mean_VS[i] = np.mean(full_sim[int(full_sim.shape[0]/2):,:,:,:])
    
#%% Plot Km effect on percentiles
fig, (ax2, ax3) = plt.subplots(1,2,figsize = (5,2.5), dpi = 400)


# Relative difference
ax2.set_xlabel("DAT K$_\mathrm{m}$ (\u00B5M)")
ax2.set_xlim(0*10**-6,6*10**-6)
ax2.set_xticks([0*10**-6,2*10**-6,4*10**-6,6*10**-6])
ax2.set_xticklabels([0,2,4,6])
ax2.set_ylabel("Fold over baseline")
ax2.set_title("Compared to no inhibition", fontsize = 10)
ax2.set_ylim(0,30)


ax2.plot(km_range[:], km_mean_VS[:]/km_mean_VS[0]+0.5, color = "cornflowerblue")
ax2.plot(km_range[:], km_mean_VS[:]/km_mean_VS[0], color = "indianred")
legend = ax2.legend(("DS","VS"), frameon = False,
            handlelength = 1.5, fontsize = 8, loc = "upper left")
legend.set_title('Mean DA',prop={'size': 8})


ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


ax3.set_title("Regional difference", fontsize = 10)
ax3.set_ylabel("VS to DS ratio")
ax3.set_xlabel("DAT K$_\mathrm{m}$ (\u00B5M)")
ax3.set_ylim(0,5)
ax3.set_xlim(0*10**-6,6*10**-6)
ax3.set_xticks([0*10**-6,2*10**-6,4*10**-6,6*10**-6])
ax3.set_xticklabels([0,2,4,6])

linestyles = ["-",":","-"]
for i in range(2):
    ax3.plot(km_range, km_percentiles_VS[2-i,:]/km_percentiles_DS[2-i,:], ls = linestyles[i], color = "k")
    
legend = ax3.legend(("Peak$^{99.5th}$","Tonic$^{50th}$"), frameon = False,
            handlelength = 1.3, prop={'size': 8}, loc = "upper right", bbox_to_anchor=[1, 1.05])
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('[DA] percentiles',prop={'size': 8})
    
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)


fig.tight_layout()
#%% Vmax dynamics at different Qs

vmax_range = np.linspace(0.5*10**-6,10*10**-6,39)
vmax_percentiles_VS_1000 = np.zeros((3,len(vmax_range)))
vmax_percentiles_VS_3000 = np.zeros((3,len(vmax_range)))
vmax_percentiles_VS_10000 = np.zeros((3,len(vmax_range)))
        
simulation_VS, space_init_VS, firing_VS, release_sites_VS, var_list_VS = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

for i, vmax_val in enumerate(vmax_range):
    print(i)
    # VS 1000
    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 1000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS_1000[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])

    # VS 3000
    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 3000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS_3000[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
    
    # VS 3000
    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 10000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS_10000[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
    


#%% Vmax dynamics at different R%

vmax_range = np.linspace(0.5*10**-6,10*10**-6,39)
vmax_percentiles_VS_3 = np.zeros((3,len(vmax_range)))
vmax_percentiles_VS_6 = np.zeros((3,len(vmax_range)))
vmax_percentiles_VS_12 = np.zeros((3,len(vmax_range)))
        
for i, vmax_val in enumerate(vmax_range):
    print(i)
    # VS 3%
    simulation_VS, space_init_VS, firing_VS, release_sites_VS, var_list_VS = \
            sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                      inter_var_distance = 25*(1/0.85), p_r = 0.03, f_rate = 4, n_neurons = 150, Hz = 0.01)

    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 3000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS_3[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
 
    # VS 6%
    simulation_VS, space_init_VS, firing_VS, release_sites_VS, var_list_VS = \
            sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                      inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 3000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS_6[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
    
    # VS 12%
    simulation_VS, space_init_VS, firing_VS, release_sites_VS, var_list_VS = \
            sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                      inter_var_distance = 25*(1/0.85), p_r = 0.30, f_rate = 4, n_neurons = 150, Hz = 0.01)

    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS, firing_VS, var_list_VS, 
                     Q = 3000, uptake_rate = vmax_val, Km = 210*10**-9, Ds = 321.7237308146399)
    
    vmax_percentiles_VS_12[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
     
#%% Vmax at different Qs

fig, (ax1, ax2) = plt.subplots(1,2,figsize = (5.5,2.7), dpi = 400, gridspec_kw={"width_ratios": [1,1]})

ax1.set_title("Effect of Q and DAT V$_{max}$ on [DA]", fontsize = 10)
ax1.set_ylabel("Relative [DA]")
ax1.set_xlabel("V$_{max}$ (\u00B5M s$^{-1}$)")
ax1.set_ylim(0,1)
ax1.set_xlim(0*10**-6,9*10**-6)
ax1.set_xticks([0*10**-6,3*10**-6,6*10**-6,9*10**-6])
ax1.set_xticklabels([0,3,6,9])
color_list = ["black","grey","lightgrey"][::-1]

DS_max_idx = np.argmin(abs(vmax_range-(4.5+4.5*0.17)*10**-6))
DS_min_idx = np.argmin(abs(vmax_range-(4.5-4.5*0.17)*10**-6))

VS_max_idx = np.argmin(abs(vmax_range-(1.5+4.5*0.17)*10**-6))
VS_min_idx = np.argmin(abs(vmax_range-(1.5-4.5*0.17)*10**-6))

# linestyles = [":","--","-"]

ax1.plot(vmax_range,(vmax_percentiles_VS_1000[1,:]/np.max(vmax_percentiles_VS_1000[1,:])),
         color = "indianred", ls = "-")
ax1.plot(vmax_range,(vmax_percentiles_VS_3000[1,:]/np.max(vmax_percentiles_VS_3000[1,:])),
         color = "indianred", ls = "--")
ax1.plot(vmax_range,(vmax_percentiles_VS_10000[1,:]/np.max(vmax_percentiles_VS_10000[1,:])), 
         color = "indianred", ls = ":")

# legend = ax1.legend(("10$^{th}$","30$^{th}$","50$^{th}$","70$^{th}$","90$^{th}$"), frameon = False,
#            handlelength = 1.2, prop={'size': 9})
legend = ax1.legend(("1000", "3000", "10000"), frameon = False,
            handlelength = 1.6, prop={'size': 8}, loc = "lower right", bbox_to_anchor = [1.02,0.05])
legend.set_title('Q',prop={'size': 8})


ax1.text(6*10**-6,0.45,
          "Median DS V$_{max}$\n\u00B1 50 %", ha = "center", fontsize = 8, rotation = 90)
ax1.fill_between(x = [3*10**-6,9*10**-6], y1 = [150,150], 
                  color = "cornflowerblue", alpha = 0.5, lw = 0)
ax1.text(2.1*10**-6,0.45,
          "Median VS V$_{max}$\n\u00B1 50 %", ha = "center", fontsize = 8, rotation = 90)
ax1.fill_between(x = [1*10**-6,3*10**-6], y1 = [150,150], 
                  color = "indianred", alpha = 0.5, lw = 0)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# With varing R%
ax2.set_title("Effect of R$_\%$ and DAT V$_{max}$ on [DA]", fontsize = 10)
ax2.set_ylabel("Relative [DA]")
ax2.set_xlabel("V$_{max}$ (\u00B5M s$^{-1}$)")
ax2.set_ylim(0,1)
ax2.set_xlim(0*10**-6,9*10**-6)
ax2.set_xticks([0*10**-6,3*10**-6,6*10**-6,9*10**-6])
ax2.set_xticklabels([0,3,6,9])

ax2.plot(vmax_range,(vmax_percentiles_VS_3[1,:]/np.max(vmax_percentiles_VS_3[1,:])),
         color = "indianred", ls = "-")
ax2.plot(vmax_range,(vmax_percentiles_VS_6[1,:]/np.max(vmax_percentiles_VS_6[1,:])),
         color = "indianred", ls = "--")
ax2.plot(vmax_range,(vmax_percentiles_VS_12[1,:]/np.max(vmax_percentiles_VS_12[1,:])), 
         color = "indianred", ls = ":")

# legend = ax1.legend(("10$^{th}$","30$^{th}$","50$^{th}$","70$^{th}$","90$^{th}$"), frameon = False,
#            handlelength = 1.2, prop={'size': 9})
legend = ax2.legend(("3%", "6%", "12%"), frameon = False,
            handlelength = 1.6, prop={'size': 8}, loc = "lower right", bbox_to_anchor = [1.02,0.05])
legend.set_title('R$_\%$',prop={'size': 8})


ax2.text(6*10**-6,0.45,
          "Median DS V$_{max}$\n\u00B1 50 %", ha = "center", fontsize = 8, rotation = 90)
ax2.fill_between(x = [3*10**-6,9*10**-6], y1 = [150,150], 
                  color = "cornflowerblue", alpha = 0.5, lw = 0)
ax2.text(2.1*10**-6,0.45,
          "Median VS V$_{max}$\n\u00B1 50 %", ha = "center", fontsize = 8, rotation = 90)
ax2.fill_between(x = [1*10**-6,3*10**-6], y1 = [150,150], 
                  color = "indianred", alpha = 0.5, lw = 0)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()