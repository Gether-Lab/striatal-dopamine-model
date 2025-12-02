#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 12:23:28 2025

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


def impulse(t,dt,b,tau):
    return b*(dt/tau)*np.exp(-t/tau)

def impulse_2(t,k1,k2,tau,ts):
    return np.exp(-(t+1)*(k1*tau+k2*ts))

def exp_decay(t,N0,time_constant):
    return N0*np.exp(-time_constant*t)

#%% Simulate burst
# DS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 100, depth = 20, dx_dy = 1, time = 3, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.001)


# Define the burst
start_time = 1 # In seconds
start_time_dt = int(start_time/var_list[1]) # convert to index

n_ap = 1 # Number of action potentials in a burst
burst_rate = 1000 # Burst firing rate (Hz)
burst_p_r = .2 # Release probability per AP during bursts


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 6*10**-6, Ds = 321.7237308146399)


sim_burst_DS = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_DS = np.mean(full_sim, axis = (1,2,3))*10**9


# VS
# # Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 100, depth = 20, dx_dy = 1, time = 3, D = 763,
                  inter_var_distance = 25*(1/0.9), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.001)


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,:] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, firing.shape[1]))

        
# Simulate the dynamics
full_sim = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 2*10**-6, Ds = 321.7237308146399)



sim_burst_VS = np.mean(full_sim, axis = (1,2,3))*10**9
sim_burst_FSCV_VS = np.mean(full_sim, axis = (1,2,3))*10**9


#%% Deconvolve and fit
rounds = 100
FSCV_all_DS = np.zeros((rounds,sim_burst_FSCV_DS[::100].shape[0]))
FSCV_all_VS = np.zeros((rounds,sim_burst_FSCV_VS[::100].shape[0]))
for i in range(rounds):
    FSCV_all_DS[i,:] = sim_burst_FSCV_DS[i::100]
    FSCV_all_VS[i,:] = sim_burst_FSCV_VS[i::100]
    
FSCV_top_DS = FSCV_all_DS[int(np.argmax(FSCV_all_DS)/(FSCV_all_DS.shape[1])),:]
FSCV_top_VS = FSCV_all_VS[int(np.argmax(FSCV_all_DS)/(FSCV_all_DS.shape[1])),:]

# Calculate FSCV response
deconvolved_signal_DS = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_DS-sim_burst_FSCV_DS[int(start_time/var_list[4]-1)], mode = "full")
deconvolved_signal_VS = ss.convolve(impulse_2(np.linspace(0,19,20),1.2,12,0.1,2*(1/60)), FSCV_top_VS-sim_burst_FSCV_VS[int(start_time/var_list[4]-1)], mode = "full")




#%% Two-pane layout

fig, (ax1, ax2) = plt.subplots(1,2,figsize = (3.1,2.5), dpi = 400, gridspec_kw={"height_ratios": [1]})
fig.suptitle("Single stimulation,\nmodelled FSCV response", fontsize = 10, y = 0.92)

# DS
ax1.plot(np.linspace(0,3,3000),sim_burst_DS-15, color = "dimgrey", ls = "-", lw = 1.2)
ax1.plot(np.linspace(0,3,30)+0.1,deconvolved_signal_DS[:-19]+3, color = "cornflowerblue", ls = "--", lw = 1.2)
ax1.legend(("Sim.", "FSCV"), title="DS", title_fontsize = 8,
           ncol = 1, handlelength = 1.3, columnspacing = 1, frameon = False,
            bbox_to_anchor = [1.05, 1], loc = "upper right", fontsize = 8)
ax1.plot([1,1],[400,380], lw = 1.5, color = "k")
ax1.text(0.94, 405, "Stim", rotation = 90, ha = "right", va = "top", fontsize = 8)

ax1.set_xlim(0.5,3)
ax1.set_xticks([1,2,3])
ax1.set_xlabel("Seconds")
ax1.set_xticklabels([0, 1, 2])
ax1.set_ylim(-20,400)
ax1.set_ylabel("\u0394[DA] (nM)")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# VS
ax2.plot(np.linspace(0,3,3000),sim_burst_VS-35, color = "dimgrey", ls = "-", lw = 1.2)
ax2.plot(np.linspace(0,3,30)+0.1,deconvolved_signal_VS[:-19], color = "firebrick", ls = "--", lw = 1.2)
ax2.legend(("Sim.", "FSCV"), title="VS", title_fontsize = 8,
           ncol = 1, handlelength = 1.3, columnspacing = 1, frameon = False,
            bbox_to_anchor = [1.05, 1], loc = "upper right", fontsize = 8)
ax2.plot([1,1],[400,380], lw = 1.5, color = "k")
ax2.text(0.94, 405, "Stim", rotation = 90, ha = "right", va = "top", fontsize = 8)

ax2.set_xlim(0.5,3)
ax2.set_xticks([1,2,3])
ax2.set_xlabel("Seconds")
ax2.set_xticklabels([0, 1, 2])
ax2.set_ylim(-20,400)
ax2.set_yticks([])
# ax2.set_ylabel("\u0394[DA] (nM)")
ax2.spines["left"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()

#%% Simulate different DS bursts

DS_single_list = np.zeros((100,10))
DS_low_list = np.zeros((100,10))
DS_med_list = np.zeros((100,10))
DS_high_list = np.zeros((100,10))


width = 100
for i in range(10):
    ############## Single
    # # Simulate release sites
    simulation, space_ph, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                      inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
            
    # Define the area
    
    r_sphere = 5
    
    ROI = (release_sites[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites[0,:] < (width/2 + r_sphere - 0.5)) & \
          (release_sites[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites[1,:] < (width/2 + r_sphere - 0.5))
          
    # Define the burst
    start_time = 0.5 # In seconds
    start_time_dt = int(start_time/var_list[1]) # convert to index
    
    n_ap = 1 # Number of action potentials in a burst
    burst_rate = 1000 # Burst firing rate (Hz)
    burst_p_r = 1 # Release probability per AP during bursts
    
    
    burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
    AP_freq = n_ap/burst_time # APs per d_t
    
    
    # Generate the burst of firing
    firing[start_time_dt:start_time_dt+burst_time,ROI] =\
        np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))
    
            
    # Simulate the dynamics
    full_sim_DS_single = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                      Q = 3000, uptake_rate = 6*10**-6, Ds = 321.7237308146399)
    
    ############## Low burst
    # # Simulate release sites
    simulation, space_ph, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                      inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
            
    # Define the area
    
    r_sphere = 5
    
    ROI = (release_sites[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites[0,:] < (width/2 + r_sphere - 0.5)) & \
          (release_sites[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites[1,:] < (width/2 + r_sphere - 0.5))
          
    # Define the burst
    start_time = 0.5 # In seconds
    start_time_dt = int(start_time/var_list[1]) # convert to index
    
    n_ap = 3 # Number of action potentials in a burst
    burst_rate = 10 # Burst firing rate (Hz)
    burst_p_r = 1 # Release probability per AP during bursts
    
    
    burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
    AP_freq = n_ap/burst_time # APs per d_t
    
    
    # Generate the burst of firing
    firing[start_time_dt:start_time_dt+burst_time,ROI] =\
        np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))
    
            
    # Simulate the dynamics
    full_sim_DS_low = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                      Q = 3000, uptake_rate = 6*10**-6, Ds = 321.7237308146399)
    
    
    ############## Med burst
    # # Simulate release sites
    simulation, space_ph, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                      inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
            
    # Define the area
    
    r_sphere = 5
    
    ROI = (release_sites[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites[0,:] < (width/2 + r_sphere - 0.5)) & \
          (release_sites[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites[1,:] < (width/2 + r_sphere - 0.5))
          
    # Define the burst
    start_time = 0.5 # In seconds
    start_time_dt = int(start_time/var_list[1]) # convert to index
    
    n_ap = 6 # Number of action potentials in a burst
    burst_rate = 20 # Burst firing rate (Hz)
    burst_p_r = 1 # Release probability per AP during bursts
    
    
    burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
    AP_freq = n_ap/burst_time # APs per d_t
    
    
    # Generate the burst of firing
    firing[start_time_dt:start_time_dt+burst_time,ROI] =\
        np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))
    
            
    # Simulate the dynamics
    full_sim_DS_med = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                      Q = 3000, uptake_rate = 6*10**-6, Ds = 321.7237308146399)
    
    
    ############## High burst
    # # Simulate release sites
    simulation, space_ph, firing, release_sites, var_list = \
            sim_space_neurons_3D(width = 100, depth = 100, dx_dy = 1, time = 1.5, D = 763,
                      inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.02)
            
    # Define the area
    
    r_sphere = 5
    
    ROI = (release_sites[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites[0,:] < (width/2 + r_sphere - 0.5)) & \
          (release_sites[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites[1,:] < (width/2 + r_sphere - 0.5))
          
    # Define the burst
    start_time = 0.5 # In seconds
    start_time_dt = int(start_time/var_list[1]) # convert to index
    
    n_ap = 12 # Number of action potentials in a burst
    burst_rate = 40 # Burst firing rate (Hz)
    burst_p_r = 1 # Release probability per AP during bursts
    
    
    burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
    AP_freq = n_ap/burst_time # APs per d_t
    
    # Generate the burst of firing
    firing[start_time_dt:start_time_dt+burst_time,ROI] =\
        np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))
    
            
    # Simulate the dynamics
    full_sim_DS_high = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                      Q = 3000, uptake_rate = 6*10**-6, Ds = 321.7237308146399)
    
    DS_single_list[:,i] = np.max(np.log10(np.mean(full_sim_DS_single[25:55,:,50,:], axis = 2)), axis = 0)
    DS_low_list[:,i] = np.max(np.log10(np.mean(full_sim_DS_low[25:55,:,50,:], axis = 2)), axis = 0)
    DS_med_list[:,i] = np.max(np.log10(np.mean(full_sim_DS_med[25:55,:,50,:], axis = 2)), axis = 0)
    DS_high_list[:,i] = np.max(np.log10(np.mean(full_sim_DS_high[25:55,:,50,:], axis = 2)), axis = 0)

#%%

# fig, (ax1, ax2) = plt.subplots(1,2,figsize = (3.5,2.5), dpi = 400, gridspec_kw={"width_ratios": [2.5,1]})
fig, (ax1, ax2) = plt.subplots(1,2,figsize = (3.5,2.5), dpi = 400, gridspec_kw={"width_ratios": [3,1.5]})
ax1.set_title("Spill-over from burst\n(40 terminals)", fontsize = 10)
ax1.set_ylabel("Peak DA after burst (\u00B5M)")
ax1.set_xlabel("Distance (\u00B5m)")
ax1.text(43,3.02, "Burst origin", rotation = 90, va = "top", ha = "right", color = "grey")


ax1.plot(10**np.mean(DS_single_list, axis = 1)*10**6, color = "lightblue", lw = 1.5, zorder = 1)
ax1.plot(10**np.mean(DS_low_list, axis = 1)*10**6, color = "cadetblue", lw = 1.5, zorder = 3)
ax1.plot(10**np.mean(DS_med_list, axis = 1)*10**6, color = "royalblue", lw = 1.5, zorder = 5)
ax1.plot(10**np.mean(DS_high_list, axis = 1)*10**6, color = "darkblue", lw = 1.5, zorder = 7)

ax1.legend(("1/10","3/10","6/20","12/40"), title = "APs/Hz", title_fontsize = 8,
           fontsize = 8, handlelength = 1, frameon = False, loc = "upper right", bbox_to_anchor = [1.1,1.02])

ax1.plot(10**np.mean(DS_single_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 0)
ax1.plot(10**np.mean(DS_low_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 2)
ax1.plot(10**np.mean(DS_med_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 4)
ax1.plot(10**np.mean(DS_high_list, axis = 1)*10**6, color = "w", lw = 2.5, zorder = 6)

ax1.fill_between([44.5,54.5], [0,0], [3,3], color = "darkgrey", zorder = -1)

ax1.set_ylim(0,3)
ax1.set_xlim(0,99)
ax1.set_xticks([0,49.5,99])
ax1.set_xticklabels([-50,0,50])

ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)


ax2.set_title("Volume\n(>100 nM)", fontsize = 10)
area_list = [(4/3*np.pi*np.sum(DS_single_list[:,:] > -7,axis = 0)**3)/(4/3*np.pi*10**3),
           (4/3*np.pi*np.sum(DS_low_list[:,:] > -7,axis = 0)**3)/(4/3*np.pi*10**3),
           (4/3*np.pi*np.sum(DS_med_list[:,:] > -7,axis = 0)**3)/(4/3*np.pi*10**3),
           (4/3*np.pi*np.sum(DS_high_list[:,:] > -7,axis = 0)**3)/(4/3*np.pi*10**3)]
color_list = ["lightblue", "cadetblue", "royalblue","darkblue"]
label_list = ["1/10","3/10","6/20","12/40"]
for i in range(4):
    ax2.scatter([np.repeat(i,10)+(np.random.rand(10)-0.5)*0.2], area_list[i],
                color = "w", edgecolor = color_list[i], s = 15)
    ax2.plot([-0.4+i,0.4+i], [np.mean(area_list[i]),np.mean(area_list[i])],
             color = color_list[i], lw = 1.5, zorder = 10)
    ax2.plot([-0.4+i,0.4+i], [np.mean(area_list[i]),np.mean(area_list[i])],
             color = "w", lw = 3.5, zorder = 9)
    
    ax2.text(i,0,label_list[i], ha = "center", va = "top", rotation = 90)

ax2.set_ylim(0,40)
ax2.set_ylabel("Relative to burst origin")
ax2.set_xlim(-0.7,3.5)
ax2.set_xticks([])


ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["bottom"].set_visible(False)

fig.tight_layout()

#%% With chain of bursts

# DS
# Simulate release sites
simulation, space_ph, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 17, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.002)

# Define the area

width = 50
r_sphere = 5
      
# Define the burst
start_time = 7 # In seconds
start_time_dt = int(start_time/var_list[1]) # convert to index

# Define the 2nd burst
start_time_2 = 7.6 # In seconds
start_time_dt_2 = int(start_time_2/var_list[1]) # convert to index

# Define the 3rd burst
start_time_3 = 8.2 # In seconds
start_time_dt_3 = int(start_time_3/var_list[1]) # convert to index

n_ap = 3 # Number of action potentials in a burst
burst_rate = 10 # Burst firing rate (Hz)
burst_p_r = 1 # Release probability per AP during bursts


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Find relevant terminals
ROI = (release_sites[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites[0,:] < (width/2 + r_sphere - 0.5)) & \
      (release_sites[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites[1,:] < (width/2 + r_sphere - 0.5))
      
# Add the burst of firing
firing[start_time_dt:start_time_dt+burst_time,ROI] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))
    
          
# Add the 2nd burst of firing
firing[start_time_dt_2:start_time_dt_2+burst_time,ROI] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))
    
# Add the 3rd burst of firing
firing[start_time_dt_3:start_time_dt_3+burst_time,ROI] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))
    
    
     
full_sim_DS_burst, occ_D1_DS_burst, occ_D2_DS_burst, occ_D1_low_DS_burst, occ_D1_high_DS_burst = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 6*10**-6, Ds = 321.7237308146399, start_occ_D2 = 0.4)

#%% Single DS plot from chain
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (2,2.5), dpi = 400)
time_y = np.linspace(-5, 10, 7500)
ax1.set_title("Chain of bursts", fontsize = 10)

site_no = 3
x_site = release_sites[0,np.where(ROI==True)[0][site_no]]
y_site = release_sites[1,np.where(ROI==True)[0][site_no]]
z_site = release_sites[2,np.where(ROI==True)[0][site_no]]


# Concentration profile
# ax1.set_title("Receptor binding", fontsize = 10)
ax1.plot(time_y, full_sim_DS_burst[1000:,x_site,y_site,z_site+1]*10**6, lw = 1, color = "cornflowerblue")

# The burst
ax1.plot([0,0.3],[3.3,3.3], color = "dimgrey", lw = 1)
ax1.plot([0.6,0.9],[3.3,3.3], color = "dimgrey", lw = 1)
ax1.plot([1.2,1.5],[3.3,3.3], color = "dimgrey", lw = 1)
ax1.text(0.75, 3.55, "3x3 APs/10 Hz", ha = "center", fontsize = 7, color = "dimgrey")


ax1.set_ylim(-0.1,2)
ax1.set_yticks([0,4])
ax1.set_xlim(-5,10)
ax1.set_xticklabels([])
ax1.set_ylabel("[DA] (\u00B5M)")


# Receptor occupany
ax2.plot(time_y, occ_D1_DS_burst[1000:,x_site,y_site,z_site+1], lw = 1, color = "teal", ls = "-")
ax2.text(-4.7,0.12, "D1", ha = "left", fontsize = 8, color = "teal")

ax2.plot(time_y, occ_D2_DS_burst[1000:,x_site,y_site,z_site+1], lw = 1, color = "royalblue", ls = "-")
ax2.text(-4.7,0.70, "D2", ha = "left", fontsize = 8, color = "royalblue")


ax2.set_ylim(-0.05,1)
ax2.set_xlim(-5,10)
ax2.set_ylabel("Occupancy")
ax2.set_xlabel("Seconds")


ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
# ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()