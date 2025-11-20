#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 20:49:08 2025

@author: ejdrup
"""

def fill_zero_concentrations_optimized(array):
    # Ensure the input is a numpy array
    array = np.array(array)

    # Check if the array is 4D
    if array.ndim != 4:
        raise ValueError("Input array must be 4-dimensional.")

    time_dim = array.shape[0]

    # Precompute the indices where array is zero
    zero_indices = np.where(array == 0)

    # Iterate only over the zero elements
    for t, x, y, z in zip(*zero_indices):
        # Find the previous non-zero concentration
        for t_prev in range(t-1, -1, -1):
            if array[t_prev, x, y, z] != 0:
                prev_conc = array[t_prev, x, y, z]
                break
        else:
            prev_conc = None

        # Find the next non-zero concentration
        for t_next in range(t+1, time_dim):
            if array[t_next, x, y, z] != 0:
                next_conc = array[t_next, x, y, z]
                break
        else:
            next_conc = None

        # Only fill in if both previous and next concentrations are found
        if prev_conc is not None and next_conc is not None:
            array[t, x, y, z] = (prev_conc + next_conc) / 2

    return array


#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv      
from cycler import cycler
from scipy import stats
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe


def points_circle(n_points = 10):
    # radius of the circle
    circle_r = 250
    # center of the circle (x, y)
    circle_x = 500
    circle_y = 500
    
    x_cords = []
    y_cords = []
    for i in range(n_points):
        # calculating coordinates
        x = circle_r*2 * (np.random.random()-0.5) + circle_x
        x_cords.append(x)
        y = circle_r*2 * (np.random.random()-0.5) + circle_y
        y_cords.append(y)
    
    return [x_cords, y_cords]

def MM_kin(c, km, vmax):
    return (vmax*c)/(km+c)

DAT_sim_conc =  np.genfromtxt("sim data/DAT_sim_conc.csv",delimiter=",")
DAT_sim_diff =  np.genfromtxt("sim data/DAT_sim_rel_time.csv",delimiter=",")
DAT_sim_dist =  np.genfromtxt("sim data/DAT_sim_dist.csv",delimiter=",")
DAT_sim_gradient =  np.genfromtxt("sim data/DAT_sim_gradient.csv",delimiter=",")
DAT_sim_conc_surface =  np.genfromtxt("sim data/DAT_sim_conc_surface.csv",delimiter=",")
DAT_sim_DA_diff =  np.genfromtxt("sim data/DAT_sim_DA_diff.csv",delimiter=",")
DAT_sim_DA_diff_3_ce =  np.genfromtxt("sim data/DAT_sim_DA_diff_3_ce.csv",delimiter=",")

custom_cycler = cycler(color=['#8B0000', '#FF0000', '#F08080', '#808080', '#000000'])

#% From newer Python simulation (same results, just different format)
# Load files
sim_result_20 = np.load("sim data/sim_result_20.npy")
sim_result_40 = np.load("sim data/sim_result_40.npy")
sim_result_80 = np.load("sim data/sim_result_80.npy")
sim_result_160 = np.load("sim data/sim_result_160.npy")
sim_result_uni = np.load("sim data/sim_result_uni.npy")


#% From newer Python simulation (same results, just different format)
# Load files
steady_result_20 = np.loadtxt("sim data/steady_state_DAT_20nm.csv")
steady_result_40 = np.loadtxt("sim data/steady_state_DAT_40nm.csv")
steady_result_80 = np.loadtxt("sim data/steady_state_DAT_80nm.csv")
steady_result_160 = np.loadtxt("sim data/steady_state_DAT_160nm.csv")
steady_result_uni = np.loadtxt("sim data/steady_state_DAT_unclus.csv")
mean_py_sim_steady = np.stack((steady_result_20,steady_result_40,steady_result_80,
                               steady_result_160,steady_result_uni)).T

# Start concentration
start_conc = 100*10**-9 # in molar

# Set first point to correct value (0 when loaded)
sim_result_20[0,:,:,:] = start_conc
sim_result_40[0,:,:,:] = start_conc
sim_result_80[0,:,:,:] = start_conc
sim_result_160[0,:,:,:] = start_conc
sim_result_uni[0,:,:,:] = start_conc

#%% Fill
# Due to unknown error in sampling method, some timepoints are not filled with data. 
# This interpolates between the previous and following concentration.
# Do reach out to the authors if you figure out the error in the sampling.
sim_result_20 = fill_zero_concentrations_optimized(sim_result_20)
sim_result_40 = fill_zero_concentrations_optimized(sim_result_40)
sim_result_80 = fill_zero_concentrations_optimized(sim_result_80)
sim_result_160 = fill_zero_concentrations_optimized(sim_result_160)
sim_result_uni = fill_zero_concentrations_optimized(sim_result_uni)


#%% Extrac mean concentrations

# Calculate their mean concs
mean_py_sim = np.zeros((401,5))
mean_py_sim[:,0] = np.mean(sim_result_20, axis = (1,2,3))
mean_py_sim[:,1] = np.mean(sim_result_40, axis = (1,2,3))
mean_py_sim[:,2] = np.mean(sim_result_80, axis = (1,2,3))
mean_py_sim[:,3] = np.mean(sim_result_160, axis = (1,2,3))
mean_py_sim[:,4] = np.mean(sim_result_uni, axis = (1,2,3))

# Relative time to 10 nm
t_rel_ten_nm = np.argmin(abs(mean_py_sim*10**9 - 10),axis = 0)/np.argmin(abs(mean_py_sim*10**9 - 10),axis = 0)[-1]



# %% Calculate distance to release site

space_radius = 10
step_size = 0.01
n_steps = int(space_radius/step_size)
space_radius = int(space_radius/step_size)
dist_center = np.zeros((n_steps, n_steps,))

# Initial conditions - circle of radius r centred at (cx,cy) (mm)
center_space = n_steps/2
for i in range(n_steps):
    for j in range(n_steps):
        dist = (i*step_size-center_space*step_size)**2 + (j*step_size-center_space*step_size)**2
        dist_center[i,j] = np.sqrt(dist)

#%% Plot the concept figure

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize = (9,2.5), dpi = 400, gridspec_kw={'width_ratios': [1,0.8,1,1]})

# Generate DA from distance to center
relative_DA = 1/(1+np.exp(-(dist_center-2)))

# Generate random points
DATs = points_circle(40)

ax1.set_title("DAT nanocluster", fontsize = 10)
# im = ax1.imshow(relative_DA, vmax = 1, cmap = "magma")
data = np.random.multivariate_normal((0, 0), [[0.8, 0.05], [0.05, 0.7]], 100)
x = np.asarray(DATs[0])
y = np.asarray(DATs[1])
xmin, xmax = 0, 1000
ymin, ymax = 0, 1000

# Peform the kernel density estimate
xx, yy = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = stats.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)
im = ax1.contourf(xx, yy, f, levels = 100, cmap='magma_r', vmin = 10**-10, vmax = 5*10**-6)
ax1.scatter(DATs[0], DATs[1], color = "w", edgecolor = "k", lw = 0.5, s = 20)
ax1.set_yticks([])
ax1.set_ylabel("200 nm")
ax1.set_xticks([])
ax1.set_xlabel("200 nm")

ax2.set_title("Uptake efficiency", fontsize = 10)
fig.text(0.375,0.83,"conceptualization", ha = "center")
DA_range = np.arange(300)
ax2.plot(MM_kin(DA_range, 210, 1500)*10**-3, color = "dimgrey")

conc_1 = 30
uptake_1 = MM_kin(conc_1, 210, 1500)*10**-3
ax2.plot([conc_1,conc_1],[0,uptake_1], color = "k", ls = "--", lw = 0.8)
ax2.plot([0,conc_1],[uptake_1,uptake_1], color = "k", ls = "--", lw = 0.8)
ax2.text(1.3,uptake_1+0.04, "Inside of\ncluster", fontsize = 8)

conc_2 = 150
uptake_2 = MM_kin(conc_2, 210, 1500)*10**-3
ax2.plot([conc_2,conc_2],[0,uptake_2], color = "k", ls = "--", lw = 0.8)
ax2.plot([0,conc_2],[uptake_2,uptake_2], color = "k", ls = "--", lw = 0.8)
ax2.text(1.3,uptake_2+0.04, "Outside of\ncluster", fontsize = 8)

ax2.set_xlim(1,1000)
ax2.set_ylim(0,1)
ax2.set_xlabel("[DA] (nM)")
ax2.set_ylabel("V$_{effective}$ (\u00B5M s$^{-1}$)")

ax2.set_xscale("log")
# ax2.set_xticks([10**0,10**1,10**2,10**3,10**4])


ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)



masked = np.ma.masked_where(DAT_sim_dist == 0, DAT_sim_dist)
a = ax3.imshow(masked, aspect = 'equal', cmap='magma_r', vmax = 1, vmin = 0.1)
ax3.set_facecolor('grey')

ax3.set_ylabel("Side (1.82 \u00B5m)", labelpad = 7)
ax3.set_xlabel("Side (1.82 \u00B5m)", labelpad = 7)
ax3.set_title("Top view", fontsize = 10)

# Legends
ax3.text(87,8, "Unfolded terminal", ha = "right", fontsize = 8)
ax3.plot([0,90],[37,37], ls = "--", color = "w", lw = 0.8)
ax3.scatter([],[], s = 40, color = "k", marker = "o", lw = 0, edgecolor = "w")
ax3.legend(("Cross-section","DAT clusters"), frameon = False, bbox_to_anchor = [1.03,0.935], loc = "upper right",
           handletextpad = 0.5, handlelength = 1.15, fontsize = 8)

ax3.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off

####### DA gradient

b = ax4.imshow(DAT_sim_gradient[-60:,:]+0.00001, aspect = 'equal', cmap='magma', vmin = 0.000175, vmax = 0.00035)
ax4.plot([0,90], [60.5,60.5], color = "grey", lw = 3)
ax4.set_ylim(61.5,0)
ax4.set_xlabel("Terminal surface (1.82 \u00B5m)", labelpad = 7)
ax4.set_ylabel("Depth (3.5 \u00B5m)", labelpad = 7)
ax4.set_title("Cross-section", fontsize = 10)

divider = make_axes_locatable(ax4)
cax = divider.append_axes("bottom", size="6%", pad=0.51)
cbar2 = fig.colorbar(b, ax = ax4, cax = cax, orientation="horizontal", ticks=[0.000175,0.00021,0.000245,0.00028,0.000315,0.00035])
# cbar2.ax.set_xticks([5*10**-5, 10*10**-5])
cbar2.ax.set_xticklabels(['0.5', '0.6', "0.7","0.8", "0.9", "1"])
cbar2.set_label('Relative [DA]', labelpad=-36)
# cbar2.outline.set_visible(False)



ax4.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off

fig.tight_layout()

cbar_ax = fig.add_axes([-0.01, 0.3, 0.008, 0.47])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel('[DA]', labelpad=-20, rotation = 90)
cbar.set_ticks([])
# #%%
# f, (ax1, ax2) = plt.subplots(1,2,figsize=(4.5, 2.5), dpi=400, gridspec_kw={'height_ratios': [1]})


#%% Plot line 2
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize = (7,2.5), dpi = 400, gridspec_kw={'width_ratios': [1,1,1,1]})

ax1.set_prop_cycle(custom_cycler)
ax1.plot(np.linspace(0,0.5,499),mean_py_sim_steady[1:-1,:], lw = 1.5)
ax1.spines['top'].set_visible (False)
ax1.spines['right'].set_visible (False)
# ax1.set_yscale("log")
ax1.set_ylim(10, 40)
ax1.set_xlim(0, 0.4)
ax1.set_xlabel("Seconds")
ax1.set_ylabel("[DA] (nM)")
ax1.set_title("Equilibrium at\nsteady state", fontsize = 10, pad = 10)
ax1.legend(("20","40","80","160", "Un."), title_fontsize = 8, ncol = 2,  title = "Cluster size (\u2300 nm)",
           loc = "upper right",frameon = False, prop={'size': 8}, bbox_to_anchor =[1.21,1.1],
           handlelength = 1.1)
ax1.plot([0,0.4], [0.01,0.01], color = "k", lw = 0.8, ls = "--", zorder = 0)


ax2.set_prop_cycle(custom_cycler)
ax2.plot(np.linspace(0,0.4,401),mean_py_sim*10**9, lw = 1.5)
ax2.spines['top'].set_visible (False)
ax2.spines['right'].set_visible (False)
ax2.set_yscale("log")
ax2.set_ylim(5, 100)
ax2.set_xlim(0, 0.4)
ax2.set_xlabel("Seconds")
ax2.set_ylabel("[DA] (nM)")
ax2.set_title("Clearance\nafter burst", fontsize = 10, pad = 10)
ax2.legend(("20","40","80","160", "Un."), title_fontsize = 8,
           loc = "upper right",frameon = False, prop={'size': 8}, bbox_to_anchor =[1.15,1.1],
           handlelength = 1.1)
ax2.plot([0,0.4], [0.01,0.01], color = "k", lw = 0.8, ls = "--", zorder = 0)



time_axis = np.linspace(0,0.4,DAT_sim_DA_diff_3_ce.shape[0])
ax3.set_prop_cycle(custom_cycler)
ax3.plot(np.linspace(0,0.4,10000), DAT_sim_DA_diff[:,:]*1000)
# ax1.set_title("Difference between\ncluster and average [DA]", fontsize = 10, pad = 10)
ax3.set_title("[DA] drop at\ncenter of cluster", fontsize = 10, pad = 10)
ax3.set_ylim(0,100)
ax3.set_ylabel("\u0394[DA] (nM)", labelpad = 0)
ax3.set_xlim(-0.02,0.4)
ax3.set_xticks([0,0.2,0.4])
ax3.set_xlabel("Seconds")
ax3.legend(("20","40","80","160", "Un."), title_fontsize = 8,
           loc = "upper right",frameon = False, prop={'size': 8}, bbox_to_anchor =[1.1,1.1],
           handlelength = 1.1)

ax3.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)


DAT_sim_DA_diff_3_ce[0,0] = 0
DAT_sim_DA_diff_3_ce[0,1] = 0


ax4.set_title("[DA] drop for\n80 nm clusters", fontsize = 10, pad = 10)
plt.plot(sim_result_80[-5,:,65,-1]/np.max(sim_result_80[-5,:,65,-1])-0.005, color = '#F08080',
         path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])
ax4.fill_between([56.5-2.5,56.5+2.5],[1,1], color = "grey", alpha = 1, lw = 0, zorder = 0)
ax4.fill_between([80.5-2.5,80.5+2.5],[1,1], color = "grey", alpha = 1, lw = 0, zorder = 0)
ax4.text(53,0.24,"Inside cluster", rotation = 90, ha = "right", va = "bottom")
ax4.text(77,0.24,"Close to cluster", rotation = 90, ha = "right", va = "bottom")
ax4.set_ylim(0.2,1)
# ax4.set_yticks([0.5,0.75,1])
ax4.set_ylabel("Relative [DA]")
ax4.set_xlim(0,90)
ax4.set_xticks([])
ax4.set_xlabel("Terminal surface\n(1.82 \u00B5m)", labelpad = 9.5)

ax4.spines["right"].set_visible(False)
ax4.spines["top"].set_visible(False)


fig.tight_layout(w_pad = 0)

#%% Uptake rates center and edge

width = DAT_sim_conc_surface.shape[0]
DAT_sim_conc_surface_list = []
for i in range(5):
    DAT_sim_conc_surface_list.append(DAT_sim_conc_surface[:,int(i*width):int((i+1)*width)])
    

fig, axes = plt.subplots(2,2,figsize=(2.3,2.2), dpi=400, gridspec_kw={'height_ratios': [1,1]})
fig.suptitle("Cluster sizes (top view)", fontsize = 10, y = 0.91, x = 0.5)

clus_titles = ["20 nm","40 nm","80 nm","160 nm","Uncl."]
for i in range(4):
    (axes.flatten())[i].set_title(clus_titles[i], fontsize = 10)
    im = (axes.flatten())[i].imshow(DAT_sim_conc_surface_list[i], aspect = 'equal', cmap='magma',
                                    vmin = np.min(DAT_sim_conc_surface_list[i])-0.0005*i,
                                    vmax = np.max(DAT_sim_conc_surface_list[i]))
    (axes.flatten())[i].tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        labelbottom=False,
        labelleft=False)

fig.tight_layout(w_pad = -0.5)

cbar_ax = fig.add_axes([0.05, 0.15, 0.03, 0.53])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel('[DA]', labelpad=-20, rotation = 90)
cbar.set_ticks([])

