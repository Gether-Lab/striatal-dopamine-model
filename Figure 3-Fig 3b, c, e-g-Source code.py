#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 22:08:25 2025

@author: ejdrup
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


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
    
    
    for i in range(int(t/dt)-1):
        
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

#%% Active release site fraction
# 0.11 terminals per um3 in DS
# https://www.sciencedirect.com/science/article/pii/0306452286902721?via%3Dihub
# (Quantification of the dopamine innervation in adult rat neostriatum)

Density_range = np.linspace(0.05,1,20)
Density_percentiles_DS = np.zeros((3,len(Density_range)))
Density_percentiles_VS = np.zeros((3,len(Density_range)))

simulation_DS, space_init_DS, firing_DS, release_sites_DS, var_list_DS = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 9, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)
        
simulation_VS, space_init_VS, firing_VS, release_sites_VS, var_list_VS = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 9*(1/0.85), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

for i, D_val in enumerate(Density_range):
    print(i)
    # DS
    # Pick random subset of "D_val" size
    subset_DS = np.random.choice(np.linspace(0,firing_DS.shape[1]-1,firing_DS.shape[1]).astype(int),
                              size = int(firing_DS.shape[1]*D_val), replace = False)
    
    full_sim = sim_dynamics_3D(simulation_DS, space_init_DS, release_sites_DS[:,subset_DS], firing_DS[:,subset_DS], var_list_DS, 
                     Q = 3000, uptake_rate = 6*10**-6, Km = 210*10**-9, Ds = 321.7237308146399)
    
    Density_percentiles_DS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])
    
    # VS
    # Pick random subset of "D_val" size
    subset_VS = np.random.choice(np.linspace(0,firing_VS.shape[1]-1,firing_VS.shape[1]).astype(int),
                              size = int(firing_VS.shape[1]*D_val), replace = False)
    
    full_sim = sim_dynamics_3D(simulation_VS, space_init_VS, release_sites_VS[:,subset_VS], firing_VS[:,subset_VS], var_list_VS, 
                     Q = 3000, uptake_rate = 2*10**-6, Km = 210*10**-9, Ds = 321.7237308146399)
    
    Density_percentiles_VS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/2):,:,:,:],[10,50,99.5])


#%% Plot active release site effect on percentiles with domanicity

fig, (ax1, ax2) = plt.subplots(1,2,figsize = (5,2.5), dpi = 400)
ax1.set_title("Active terminals", fontsize = 10)
ax1.set_ylabel("[DA] (M)")
ax1.set_xlabel("Percentage active")
ax1.set_yscale("log")
ax1.set_ylim(10**-9,5*10**-7)
ax1.set_xlim(0,100)
color_list = ["black","grey","lightgrey"][::-1]

# "Fake" legends
ax1.plot([],[], color = "k", ls = "-")
# ax1.plot([],[], color = "k", ls = "-.")
# ax1.plot([],[], color = "k", ls = "--")
ax1.plot([],[], color = "k", ls = ":")
legend = ax1.legend(("Peak$^{99.5th}$","Tonic$^{50th}$"), frameon = False,
            # handlelength = 1.5, prop={'size': 8}, loc = "lower right",)
            handlelength = 1.2, prop={'size': 8}, bbox_to_anchor=[1.1, 0.36], loc = "upper right")
legend.set_title('Dopamine levels',prop={'size': 8})
# fig.text(0.359+0.042,0.52,"DS", color = "cornflowerblue", fontsize = 8)
# fig.text(0.435,0.52,"/", color = "k", fontsize = 8)
# fig.text(0.445,0.52,"VS", color = "indianred", fontsize = 8)

# Conventional Km indicator
# ax1.vlines(210*10**-9,0,300, color = "dimgrey", lw = 0.8, ls = "-")
# ax1.text(350*10**-9, 300, "Conventional K$_\mathrm{m}$",rotation = 90, ha = "left", va = "top", color = "dimgrey")

linestyles = [":",":","-"]
# DS and VS lines
for i in range(2):
    ax1.plot(Density_range*100,Density_percentiles_DS[i+1,:], color = "cornflowerblue", ls = linestyles[i+1])
    ax1.plot(Density_range*100,Density_percentiles_VS[i+1,:], color = "indianred", ls = linestyles[i+1])

    
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# Relative difference
ax2.set_xlabel("Percentage active")
ax2.set_title("Focality of dopamine", fontsize = 10)
ax2.set_ylabel("99.5$^{th}$/50$^{th}$ percentile")
ax2.set_ylim(0,60)
ax2.set_xlim(0,100)


ax2.plot(Density_range*100,Density_percentiles_DS[2,:]/Density_percentiles_DS[1,:], color = "cornflowerblue")
ax2.plot(Density_range*100,Density_percentiles_VS[2,:]/Density_percentiles_VS[1,:], color = "indianred")

legend = ax2.legend(("DS","VS"), frameon = False,
            # handlelength = 1.5, prop={'size': 8}, loc = "lower right",)
            handlelength = 1.2, prop={'size': 8}, bbox_to_anchor=[1.05, 1], loc = "upper right")

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()

#%% Q akin to the Vmax range testing

Q_range = np.linspace(1000,30000,30) # 30 for main figure, 291 for supp
Q_percentiles_DS = np.zeros((4,len(Q_range)))
Q_percentiles_VS = np.zeros((4,len(Q_range)))
mean_DS = np.zeros((len(Q_range),))
mean_VS = np.zeros((len(Q_range),))

simulation, space_init, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25, p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

for i, Q_val in tqdm(enumerate(Q_range)):
    # print(i)
    full_sim = sim_dynamics_3D(simulation, space_init, release_sites, firing, var_list, 
                     Q = Q_val, uptake_rate = 6*10**-6, Km = 210*10**-9, Ds = 321.7237308146399)
    
    Q_percentiles_DS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/4):,:,:,:],[10,50,99.5,90])
    mean_DS[i] = np.mean(full_sim[int(full_sim.shape[0]/4):,:,:,:])
    
simulation, space_init, firing, release_sites, var_list = \
        sim_space_neurons_3D(width = 50, depth = 50, dx_dy = 1, time = 2, D = 763,
                  inter_var_distance = 25*(1/0.9), p_r = 0.06, f_rate = 4, n_neurons = 150, Hz = 0.01)

for i, Q_val in tqdm(enumerate(Q_range)):
    # print(i)
    full_sim = sim_dynamics_3D(simulation, space_init, release_sites, firing, var_list, 
                     Q = Q_val, uptake_rate = 2*10**-6, Km = 210*10**-9, Ds = 321.7237308146399)
    
    Q_percentiles_VS[:,i] = np.percentile(full_sim[int(full_sim.shape[0]/4):,:,:,:],[10,50,99.5,90])
    mean_VS[i] = np.mean(full_sim[int(full_sim.shape[0]/4):,:,:,:])

#%% Plot Q effect on percentiles v2
fig, (ax1, ax3, ax2) = plt.subplots(1,3,figsize = (7.5,2.5), dpi = 400)
ax1.set_title("Effect of Q on DA levels", fontsize = 10)
ax1.set_ylabel("[DA] (nM)")
ax1.set_xlabel("Q (DA molecules)")
ax1.set_ylim(0,1500)
ax1.set_xlim(0,30000)
color_list = ["black","grey","lightgrey"][::-1]

# "Fake" legends
ax1.plot([],[], color = "k", ls = "-")
# ax1.plot([],[], color = "k", ls = "-.")
ax1.plot([],[], color = "k", ls = ":")
legend = ax1.legend(("Peak$^{99.5th}$","Tonic$^{50th}$"), frameon = False,
            handlelength = 1.2, prop={'size': 8}, loc = "upper left")
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('[DA] percentiles',prop={'size': 8})

# Conventional Km indicator
# ax1.vlines(210*10**-9,0,300, color = "dimgrey", lw = 0.8, ls = "-")
# ax1.text(350*10**-9, 300, "Conventional K$_\mathrm{m}$",rotation = 90, ha = "left", va = "top", color = "dimgrey")

linestyles = [":","-","-"]
# DS and VS lines
for i in range(2):
    ax1.plot(Q_range,Q_percentiles_DS[i+1,:]*10**9, color = "cornflowerblue", ls = linestyles[i])
    if i == 10:
        ax1.plot(Q_range+80,Q_percentiles_VS[i+1,:]*10**9, color = "indianred", ls = linestyles[i])
    else:
        ax1.plot(Q_range,Q_percentiles_VS[i+1,:]*10**9, color = "indianred", ls = linestyles[i])

    
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# Relative difference
ax2.set_xlabel("Q (DA molecules)")
ax2.set_title("Regional differences", fontsize = 10)
ax2.set_ylabel("VS to DS ratio")
ax2.set_ylim(0,10)
ax2.set_xlim(0,30000)

linestyles = [":",":","-"]
# DS and VS relative line
ax2.plot(Q_range[:],(Q_percentiles_VS[2,:]/Q_percentiles_DS[2,:]), color = "k", ls = linestyles[2])
ax2.plot(Q_range[:],(Q_percentiles_VS[1,:]/Q_percentiles_DS[1,:]), color = "k", ls = linestyles[1])

legend = ax2.legend(("Peak$^{99.5th}$","Tonic$^{50th}$"), frameon = False,
            handlelength = 1.2, prop={'size': 8}, loc = "upper left", ncol = 1, columnspacing = 0.7, bbox_to_anchor = [0,1])
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('[DA] percentiles',prop={'size': 8})

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


# Relative difference
ax3.set_xlabel("Q (DA molecules)")
ax3.set_title("Focality of dopamine", fontsize = 10)
ax3.set_ylabel("99.5$^{th}$/50$^{th}$ percentile")
ax3.set_ylim(0,20)
ax3.set_xlim(0,30000)
ax3.plot(Q_range[:],Q_percentiles_DS[2,:]/Q_percentiles_DS[1,:], color = "cornflowerblue")
ax3.plot(Q_range[:],Q_percentiles_VS[2,:]/Q_percentiles_VS[1,:], color = "indianred")

legend = ax3.legend(("DS", "VS"), frameon = False,
            handlelength = 1.2, prop={'size': 8}, loc = "upper right")
            # handlelength = 1.2, prop={'size': 9}, bbox_to_anchor=[0.38, 1.02], loc = "upper right")
legend.set_title('Region',prop={'size': 8})


ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

fig.tight_layout()