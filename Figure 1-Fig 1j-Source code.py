#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 18:54:09 2025

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
    
    occ_D1_low = space0.copy()
    occ_D1_ph_low = space_ph.copy()
    EC50_D1_low = 100*10**-9
    k_off_D1_low = (EC50_D1_low+12.4*10**-9)/(52*10**-9)
    k_on_D1_low = k_off_D1_low/EC50_D1_low
    
    occ_D1_high = space0.copy()
    occ_D1_ph_high = space_ph.copy()
    EC50_D1_high = 10000*10**-9
    k_off_D1_high = (EC50_D1_high+12.4*10**-9)/(52*10**-9)
    k_on_D1_high = k_off_D1_high/EC50_D1_high
    
    
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
        
        # D1 Occ low
        d_recep_occ = space_ph*k_on_D1_low*(1-occ_D1_ph_low) - k_off_D1_low*occ_D1_ph_low
        occ_D1_ph_low = occ_D1_ph_low + dt*d_recep_occ
        
        # D1 Occ high
        d_recep_occ = space_ph*k_on_D1_high*(1-occ_D1_ph_high) - k_off_D1_high*occ_D1_ph_high
        occ_D1_ph_high = occ_D1_ph_high + dt*d_recep_occ
        
        # Save snapshot at specified Hz
        if i%int(Hz/dt) == 0:
            space0[int(i/(Hz/dt)),:,:,:] = space_ph
            #Save receptor occupany
            occ_D2[int(i/(Hz/dt)),:,:,:] = occ_D2_ph
            
            #Save receptor occupany
            occ_D1[int(i/(Hz/dt)),:,:,:] = occ_D1_ph
            
            #Save receptor occupany
            occ_D1_low[int(i/(Hz/dt)),:,:,:] = occ_D1_ph_low
            
            #Save receptor occupany
            occ_D1_high[int(i/(Hz/dt)),:,:,:] = occ_D1_ph_high
            
        
    return space0, occ_D1, occ_D2, occ_D1_high, occ_D1_low


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
     
full_sim_DS_burst, occ_D1_DS_burst, occ_D2_DS_burst, occ_D1_low_DS_burst, occ_D1_high_DS_burst = sim_dynamics_3D(simulation, space_ph, release_sites, firing, var_list, 
                  Q = 3000, uptake_rate = 6*10**-6, Ds = 321.7237308146399, start_occ_D2 = 0.4)

#%% Single DS plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (2,2.5), dpi = 400)
time_y = np.linspace(-5, 10, 7500)
ax1.set_title("Burst and occupancy", fontsize = 10)

site_no = 11
x_site = release_sites[0,np.where(ROI==True)[0][site_no]]
y_site = release_sites[1,np.where(ROI==True)[0][site_no]]
z_site = release_sites[2,np.where(ROI==True)[0][site_no]]


# Concentration profile
# ax1.set_title("Receptor binding", fontsize = 10)
ax1.plot(time_y, full_sim_DS_burst[1000:,x_site,y_site,z_site]*10**6, lw = 1, color = "cornflowerblue")

# The burst
ax1.plot([0,0.3],[6.5,6.5], color = "dimgrey")
ax1.text(1, 7, "6 APs/20 Hz", ha = "center", fontsize = 10, color = "dimgrey")

ax1.set_ylim(-0.1,2)
ax1.set_yticks([0,8])
ax1.set_xlim(-5,10)
ax1.set_xticklabels([])
ax1.set_ylabel("[DA] (\u00B5M)")


# Receptor occupany
ax2.plot(time_y, occ_D1_DS_burst[1000:,x_site,y_site,z_site], lw = 1, color = "teal", ls = "-")
ax2.text(-4.4,0.12, "D1", ha = "left", fontsize = 10, color = "teal")

ax2.plot(time_y, occ_D2_DS_burst[1000:,x_site,y_site,z_site], lw = 1, color = "royalblue", ls = "-")
ax2.text(-4.4,0.66, "D2", ha = "left", fontsize = 10, color = "royalblue")


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
#%% DR off-rate 

fig, ax1 = plt.subplots(figsize = (2.8,2.5), dpi = 400)
ax1.set_title("Off-rate of D1R", fontsize = 10)

conc = np.mean(full_sim_DS_burst[1000:,20:30,20:30,20:30]*10**6, axis = (1,2,3))
ax1.plot(time_y, conc/np.max(conc)*1.15, lw = 1, color = "cornflowerblue")
ax1.set_ylim(0,1.5)
ax1.set_yticks([0,0.5,1, 1.5])
ax1.set_ylabel('Normalized [DA]', color='cornflowerblue')
ax1.set_xlabel("Seconds from burst")
ax1.fill_between([0,0.3],[1.5,1.5], color = "grey", zorder = 0, alpha = 0.5, lw = 0)
ax1.text(0.15,1.37, "Burst", ha = "center", fontsize = 8)


# Occupancy y-axis
ax2 = ax1.twinx()
occ = np.mean(occ_D1_DS_burst[1000:,20:30,20:30,20:30], axis = (1,2,3))
occ_low = np.mean(occ_D1_low_DS_burst[1000:,20:30,20:30,20:30], axis = (1,2,3))
occ_high = np.mean(occ_D1_high_DS_burst[1000:,20:30,20:30,20:30], axis = (1,2,3))
ax2.plot(time_y, occ_low/np.max(occ_low), lw = 1, color = "teal", ls = "--")
ax2.plot(time_y, occ/np.max(occ), lw = 1, color = "teal", ls = "-")
ax2.plot(time_y, occ_high/np.max(occ_high), lw = 1, color = "teal", ls = ":")
ax2.set_xlim(-0.2,1)
ax2.set_xticks([0,0.5,1])
ax2.set_ylim(0,1.5)
ax2.set_yticks([0,0.5,1,1.5])
ax2.set_ylabel('Normalized D1 occupancy', color='teal')
ax2.legend(("10 \u00B5M","1.0 \u00B5M","0.1 \u00B5M"), 
            frameon = False, loc = "upper right", ncol = 1, handlelength = 1.1, handletextpad = 0.5,
            columnspacing = 0.5,fontsize = 8, title="Affinities", title_fontsize = 8)

# Plot when below 0.5
time_DA_decay = np.argwhere((conc/np.max(conc)*1.15)[2500+150:] < 0.5)[0]
time_D1_decay = np.argwhere((occ/np.max(occ))[2500+150:] < 0.5)[0]

# Plot when below 0.5
time_DA_decay = np.argwhere((conc/np.max(conc)*1.15)[2500+150:] < 0.5)[0]
time_D1_low_decay = np.argwhere((occ_low/np.max(occ_low))[2500+150:] < 0.5)[0]
time_D1_high_decay = np.argwhere((occ_high/np.max(occ_high))[2500+150:] < 0.48)[0]

ax2.scatter([time_DA_decay/500+0.3], [0.51], s = 3, zorder = 10, color = "k")
ax2.scatter([time_D1_decay/500+0.3], [0.51], s = 3, zorder = 10, color = "k")
ax1.plot([time_DA_decay/500+0.3,time_D1_decay/500+0.3], [0.51,0.51], lw = 1, zorder = -10, color = "k")
ax2.text(0.4,0.54, "54 ms", ha = "left", va = "center", fontsize = 7)

ax2.scatter([time_DA_decay/500+0.3], [0.54], s = 3, zorder = 10, color = "k")
ax2.scatter([time_D1_low_decay/500+0.3], [0.54], s = 3, zorder = 10, color = "k")
ax1.plot([time_DA_decay/500+0.3,time_D1_low_decay/500+0.3], [0.54,0.54], lw = 1, zorder = -10, color = "k")
ax2.text(0.11,0.54, "2 ms", ha = "left", va = "center", fontsize = 7)

ax2.scatter([time_DA_decay/500+0.3], [0.48], s = 3, zorder = 10, color = "k")
ax2.scatter([time_D1_high_decay/500+0.3], [0.48], s = 3, zorder = 10, color = "k")
ax1.plot([time_DA_decay/500+0.3,time_D1_high_decay/500+0.3], [0.48,0.48], lw = 1, zorder = -10, color = "k")
ax2.text(0.69,0.54, "410 ms", ha = "left", va = "center", fontsize = 7)


ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

fig.tight_layout()

