#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 21:04:28 2025

@author: ejdrup
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.signal as ss
from scipy.optimize import curve_fit


def sim_space_neurons_3D(width = 100, depth = 10, dx_dy = 1, time = 1, D = 763, 
              inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01):
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


    return space0, space_ph, firing.T, np.array([x_varico,y_varico, z_varico]), np.array([time, dt, dx_dy, inter_var_distance, Hz])



def sim_dynamics_3D(space0, space_ph, release_sites, firing, var_list, 
                 Q = 3000, uptake_rate = 4*10**-6, Km = 210*10**-9,
                 Ds = 321.7237308146399, ECF = 0.21, start_occ_D2 = 0.3):
    # print(uptake_rate)
    # print(Q)
    # Extract parameters
    t = var_list[0]
    dt = var_list[1]
    dx_dy = var_list[2]
    Hz = var_list[4]
    
    # Test for receptor occupany
    # Affinities D2: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8071183/
    occ_D1 = space0.copy()
    occ_D1_ph = space_ph.copy()
    EC50_D1 = 1000*10**-9
    k_off_D1 = (EC50_D1+12.4*10**-9)/(52*10**-9)
    k_on_D1 = k_off_D1/EC50_D1
    
    
    occ_D2 = space0.copy()
    occ_D2_ph = space_ph.copy()
    occ_D2_ph[:,:,:] = start_occ_D2
    EC50_D2 = 7*10**-9
    k_off_D2 = 0.2
    k_on_D2 = k_off_D2/EC50_D2
    
    
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
        # D2 Occ        
        d_recep_occ = space_ph*k_on_D2*(1-occ_D2_ph) - k_off_D2*occ_D2_ph
        occ_D2_ph = occ_D2_ph + dt*d_recep_occ
        
        # D1 Occ
        d_recep_occ = space_ph*k_on_D1*(1-occ_D1_ph) - k_off_D1*occ_D1_ph
        occ_D1_ph = occ_D1_ph + dt*d_recep_occ
        
        # Save snapshot at specified Hz
        if i%int(Hz/dt) == 0:
            space0[int(i/(Hz/dt)),:,:,:] = space_ph
            #Save receptor occupany
            occ_D2[int(i/(Hz/dt)),:,:,:] = occ_D2_ph
            
            #Save receptor occupany
            occ_D1[int(i/(Hz/dt)),:,:,:] = occ_D1_ph
            
        
    return space0, occ_D1, occ_D2


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

def impulse_2(t,k1,k2,tau,ts):
    return np.exp(-(t+1)*(k1*tau+k2*ts))

def amp_impulse(t,tau):
    t = t*0.1
    return (-1)**0/(2*t+1)*np.exp(-(2*t+1)**2*t/tau)

def exp_decay(t,N0,time_constant):
    return N0*np.exp(-time_constant*t)

#%% With burst

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

n_ap = 6 # Number of action potentials in a burst
burst_rate = 20 # Burst firing rate (Hz)
burst_p_r = 1 # Release probability per AP during bursts


burst_time = int(1/var_list[1]*(n_ap/burst_rate)) # Length of the burst
AP_freq = n_ap/burst_time # APs per d_t

# Find relevant terminals
ROI = (release_sites[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites[0,:] < (width/2 + r_sphere - 0.5)) & \
      (release_sites[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites[1,:] < (width/2 + r_sphere - 0.5))
      
# Add the burst of firing
firing[start_time_dt:start_time_dt+burst_time,ROI] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))
     
full_sim_DS_burst, occ_D1_DS_burst, occ_D2_DS_burst = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 6*10**-6, Ds = 321.7237308146399, start_occ_D2 = 0.4)

site_no = 2
x_site = release_sites[0,np.where(ROI==True)[0][site_no]]
y_site = release_sites[1,np.where(ROI==True)[0][site_no]]
z_site = release_sites[2,np.where(ROI==True)[0][site_no]]

#%% VS
# Simulate release sites
simulation, space_ph, firing, release_sites_VS, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 17, D = 763,
                  inter_var_distance = 25*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.002)

# Find relevant terminals
ROI = (release_sites_VS[0,:] > (width/2 - r_sphere - 0.5)) & (release_sites_VS[0,:] < (width/2 + r_sphere - 0.5)) & \
      (release_sites_VS[1,:] > (width/2 - r_sphere - 0.5)) & (release_sites_VS[1,:] < (width/2 + r_sphere - 0.5))
      
# Generate the burst of firing
firing[start_time_dt:start_time_dt+burst_time,ROI] =\
    np.random.poisson(AP_freq * burst_p_r, (burst_time, np.sum(ROI)))

full_sim_VS_burst, occ_D1_VS_burst, occ_D2_VS_burst = sim_dynamics_3D(simulation, space_ph, release_sites_VS, firing, var_list, 
                  Q = 3000, uptake_rate = 2*10**-6, Ds = 321.7237308146399, start_occ_D2 = 0.75)

# %% DS-VS burst plot

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (2.4,2.5), dpi = 400)
time_y = np.linspace(-5, 10, 7500)

site_no = 2
x_site_VS = release_sites_VS[0,np.where(ROI==True)[0][site_no]]
y_site_VS = release_sites_VS[1,np.where(ROI==True)[0][site_no]]
z_site_VS = release_sites_VS[2,np.where(ROI==True)[0][site_no]]


# Concentration profile
# ax1.set_title("Receptor binding", fontsize = 10)
ax1.plot(time_y, full_sim_VS_burst[1000:,x_site_VS,y_site_VS,z_site_VS+1]*10**6, lw = 1, color = "indianred")

# The burst
ax1.plot([0,0.3],[3.3,3.3], color = "dimgrey")
ax1.text(0.15, 3.55, "6 APs/20 Hz", ha = "center", fontsize = 8, color = "dimgrey")

ax1.set_ylim(-0.1,2)
ax1.set_yticks([0,4])
ax1.set_xlim(-5,10)
ax1.set_xticklabels([])
ax1.set_ylabel("[DA] (\u00B5M)")

# Receptor occupany
# VS
ax2.plot(time_y, occ_D1_VS_burst[1000:,x_site_VS,y_site_VS,z_site_VS+1], lw = 1, color = "lightcoral", ls = "-")
ax2.text(-4.7,0.115, "D1", ha = "left", fontsize = 8, color = "lightcoral")

ax2.plot(time_y, occ_D2_VS_burst[1000:,x_site_VS,y_site_VS,z_site_VS+1], lw = 1, color = "firebrick", ls = "-")
ax2.text(-4.7,0.85, "D2", ha = "left", fontsize = 8, color = "firebrick")

# DS
ax2.plot(time_y, occ_D1_DS_burst[1000:,x_site,y_site,z_site+1], 
         lw = 1, color = "teal", ls = ":", zorder = 0, alpha = 0.5)
# ax2.text(-4.7,0.08, "D1", ha = "left", fontsize = 8, color = "teal")

ax2.plot(time_y, occ_D2_DS_burst[1000:,x_site,y_site,z_site+1], 
         lw = 1, color = "royalblue", ls = ":", zorder = 0, alpha = 0.5)
# ax2.text(-4.7,0.47, "D2", ha = "left", fontsize = 8, color = "royalblue")

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

#%% Max to min graph

fig, ax1 = plt.subplots(1, 1, figsize = (2.1,2.5), dpi = 400)
ax1.set_title("Relative occupany", fontsize = 10)
ax1.set_ylabel("Relative occupany")
ax1.set_xlabel("Distance (\u00B5m)")
ax1.text(-2,0.5, "Burst area", ha = "center", va = "center", rotation = 90, color = "dimgrey")
ax1.set_ylim(-0.04,1)
ax1.set_yticks([0,1])
ax1.set_xlim(-4,20)

x_range = np.linspace(-4,20,24)


max_D1_DS = np.max(np.mean(occ_D1_DS_burst[1000:,25,25:-1,:],axis = 2), axis = 0)
max_D2_DS = np.max(np.mean(occ_D2_DS_burst[1000:,25,25:-1,:],axis = 2), axis = 0)

max_D1_VS = np.max(np.mean(occ_D1_VS_burst[1000:,25,25:-1,:],axis = 2), axis = 0)
max_D2_VS = np.max(np.mean(occ_D2_VS_burst[1000:,25,25:-1,:],axis = 2), axis = 0)


ax1.plot(x_range, (max_D1_DS-np.min(max_D1_DS))/np.max(max_D1_DS-np.min(max_D1_DS)), 
         color = "royalblue", ls = "-", clip_on = False)
ax1.plot(x_range, (max_D2_DS-np.min(max_D2_DS))/np.max(max_D2_DS-np.min(max_D2_DS)), 
         color = "lightskyblue", ls = "-", clip_on = False)

ax1.plot(x_range, (max_D1_VS-np.min(max_D1_VS))/np.max(max_D1_VS-np.min(max_D1_VS)), 
         color = "firebrick", ls = "-", clip_on = False)
ax1.plot(x_range, (max_D2_VS-np.min(max_D2_VS))/np.max(max_D2_VS-np.min(max_D2_VS)), 
         color = "lightcoral", ls = "-", clip_on = False)

ax1.legend(("", "", "D1", "D2"), ncol = 2, fontsize = 8, handlelength = 1.2, frameon = False,
           columnspacing = 0.2, loc = "upper right", bbox_to_anchor = [1,0.94])
ax1.text(6.8, 0.9, "DS", fontsize = 8)
ax1.text(11.2, 0.9, "VS", fontsize = 8)

ax1.fill_between([-4,0], [-1,-1], [1,1], color = "grey", alpha = 0.4, lw = 0)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()