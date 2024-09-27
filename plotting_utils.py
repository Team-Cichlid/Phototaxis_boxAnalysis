import numpy as np
import matplotlib.pyplot as plt # plotting library
import pandas as pd # read and tidy up data
import os
import json # useful for loading json files (such as the ROI file)
import datetime as dt # useful for doing datetime arithmetic (subtracting days from each other)
from scipy.stats import gaussian_kde
from time import time
from glob import glob

# Set figure size in inches (width, height)
FIGURE_WIDTH = 20
FIGURE_HEIGHT = 5

# Set plotting defaults
DEFAULT_ALPHA = 0.3
LINE_WIDTH = 0.5

# Set chamber colors in RGB format
ENTRANCE_COLOR = [0., 0.74117647, 0.76862745]
LAYING_COLOR = [0.77647059, 0.48627451, 1.]
DEEP_COLOR = [0.48627451, 0.68235294, 0.]


def save_figure(fig, fig_path):
    # get current timestamp, which is used as part of the saved figure's file name (to avoid overwriting the same file)
    fig.savefig(fig_path, bbox_inches='tight')
    print('Figure saved to:', os.path.abspath(fig_path))
    return

def get_dpf_separators(dataframe):
    # getting dpf x ticks
    min_time_per_dpf = dataframe.groupby('dpf')['time_of_day'].min()
    xtick_positions = []
    for timestamp in min_time_per_dpf:
        xtick_positions.append(np.where(dataframe['time_of_day'] == timestamp)[0][0])
    xtick_labels = np.sort(dataframe['dpf'].unique())
    print('xtick positions:\t', xtick_positions)
    print('xtick labels:\t\t', xtick_labels)
    return xtick_positions, xtick_labels

def plot_chamberplot(dataframe, color_matrix, y_labels, figsize=(25, 5)):
    xtick_positions, xtick_labels = get_dpf_separators(dataframe)
    fig = plt.figure(figsize=figsize)
    plt.imshow(color_matrix, aspect='auto', interpolation='nearest')
    plt.xticks(xtick_positions, xtick_labels)
    for x in xtick_positions:
        plt.axvline(x, color='grey', linestyle="--")
    plt.xlabel("days post fertilization")
    plt.xlim(0,)

    plt.yticks([0,1,2], y_labels)

    plt.title('Chamber Activity')
    return fig

def plot_vertical_line_plot(dataframe, full_dataframe, max_dpf=None):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharex=True)
    
    entrance_df = dataframe[dataframe['entrance']!=0]
    for _, i in entrance_df.iterrows():
        axs.vlines(i['time_of_day'], ymin=2, ymax=3, color=ENTRANCE_COLOR, alpha=i['entrance'], linewidth=LINE_WIDTH)
    
    laying_df = dataframe[dataframe['laying']!=0]
    for _, i in laying_df.iterrows():
        axs.vlines(i['time_of_day'], ymin=1, ymax=2, color=LAYING_COLOR, alpha=i['laying'], linewidth=LINE_WIDTH)
    
    deep_df = dataframe[dataframe['deep']!=0]
    for _, i in deep_df.iterrows():
        axs.vlines(i['time_of_day'], ymin=0, ymax=1, color=DEEP_COLOR, alpha=i['deep'], linewidth=LINE_WIDTH)
    
    #TODO extrapolate backwards in time to 0 dpf
    dpf_positions = full_dataframe.groupby(['dpf'])['time_of_day'].min().sort_values()
    dashed_lines = []
    for dpf in dpf_positions:
        dashed_lines.append(dpf.replace(hour=0, minute=0, second=0))

    # Minimum x axis adjustment
    min_dash = dashed_lines[0]
    for i in range(0, full_dataframe['dpf'].min()):
        dashed_lines.insert(0, min_dash - dt.timedelta(hours=24*(i + 1)))
    
    # Maximum x axis adjustment
    if max_dpf is not None:
        assert max_dpf >= full_dataframe['dpf'].max(), "Hey Ash, max_dpf should be greater than the dataframe's maximum dpf"
        max_dash = dashed_lines[-1]
        for i in range(0, (max_dpf - full_dataframe['dpf'].max())):
            dashed_lines.append(max_dash + dt.timedelta(hours=24*(i + 1)))
    dashed_lines.sort()

    # Add vertical dashed lines at each dpf x-tick
    for lines in dashed_lines:
        axs.axvline(lines, color='grey', linestyle="--", linewidth=1)
    xticks = []
    for dpf in dashed_lines:
        xticks.append(dpf.replace(hour=12, minute=0, second=0))
    axs.set_xticks(xticks)
    axs.set_xticklabels(range(0, max_dpf+1) if max_dpf is not None else range(0, full_dataframe['dpf'].max()+1))
    axs.tick_params(axis=u'both', which=u'both',length=0)

    axs.set_xlim(left=dashed_lines[0], right=dashed_lines[-1].replace(hour=23, minute=59, second=59))
    axs.set_xlabel('Days Post Fertilization (dpf)')
    axs.set_yticks([0.5,1.5,2.5])
    axs.set_yticklabels(['deep chamber', 'laying chamber', 'entrance chamber'])
    axs.set_ylim(0,3)
    
    return fig

def plot_vertical_line_plot_light_dark(dataframe, full_dataframe, max_dpf=None):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharex=True)
    
    entrance_df = dataframe[dataframe['deep_one']!=0]
    for _, i in entrance_df.iterrows():
        axs.vlines(i['time_of_day'], ymin=2, ymax=3, color=ENTRANCE_COLOR, alpha=i['deep_one'], linewidth=LINE_WIDTH)
    
    laying_df = dataframe[dataframe['deep_two']!=0]
    for _, i in laying_df.iterrows():
        axs.vlines(i['time_of_day'], ymin=1, ymax=2, color=LAYING_COLOR, alpha=i['deep_two'], linewidth=LINE_WIDTH)
    
    deep_df = dataframe[dataframe['deep_three']!=0]
    for _, i in deep_df.iterrows():
        axs.vlines(i['time_of_day'], ymin=0, ymax=1, color=DEEP_COLOR, alpha=i['deep_three'], linewidth=LINE_WIDTH)
    
    #extrapolate backwards in time to 0 dpf
    dpf_positions = full_dataframe.groupby(['dpf'])['time_of_day'].min().sort_values()
    dashed_lines = []
    for dpf in dpf_positions:
        dashed_lines.append(dpf.replace(hour=0, minute=0, second=0))

    # Minimum x axis adjustment
    min_dash = dashed_lines[0]
    for i in range(0, full_dataframe['dpf'].min()):
        dashed_lines.insert(0, min_dash - dt.timedelta(hours=24*(i + 1)))
    
    # Maximum x axis adjustment
    if max_dpf is not None:
        assert max_dpf >= full_dataframe['dpf'].max(), "Hey Ash, max_dpf should be greater than the dataframe's maximum dpf"
        max_dash = dashed_lines[-1]
        for i in range(0, (max_dpf - full_dataframe['dpf'].max())):
            dashed_lines.append(max_dash + dt.timedelta(hours=24*(i + 1)))
    dashed_lines.sort()

    # Add vertical dashed lines at each dpf x-tick
    for lines in dashed_lines:
        axs.axvline(lines, color='grey', linestyle="--", linewidth=1)
    xticks = []
    for dpf in dashed_lines:
        xticks.append(dpf.replace(hour=12, minute=0, second=0))
    axs.set_xticks(xticks)
    axs.set_xticklabels(range(0, max_dpf+1) if max_dpf is not None else range(0, full_dataframe['dpf'].max()+1))
    axs.tick_params(axis=u'both', which=u'both',length=0)

    axs.set_xlim(left=dashed_lines[0], right=dashed_lines[-1].replace(hour=23, minute=59, second=59))
    axs.set_xlabel('Days Post Fertilization (dpf)')
    axs.set_yticks([0.5,1.5,2.5])
    axs.set_yticklabels(['deep_three','deep_two','deep_one'])
    axs.set_ylim(0,3)
    
    return fig

# returns a list of (x,y) coordinates which fall in ANY of the ROIs described by ROI_dict
def get_xy_coords_in_ROIs(width, height, ROI_dict):
    # get a list of the (x,y) coordinates which are in the ROIs.
    xs, ys = np.meshgrid(np.arange(0, width), np.arange(0, height))
    xy_coords = np.array([xs.ravel(), ys.ravel()]).T
    
    # check which indices are in any of the ROIs
    mask = np.zeros(len(xy_coords), dtype=bool)
    for _, roi_polygon in ROI_dict.items():
        tmp_mask = roi_polygon.contains_points(xy_coords)
        mask = np.logical_or(mask, tmp_mask)
    xy_coords_in_roi = xy_coords[mask]
    return xy_coords_in_roi

def fit_kde_model_per_dpf(dataframe, width, height, ROI_dict, kde_folder_path, density_bandwidth):
    # get a list of the (x,y) coordinates which are in the ROIs.
    xy_coords_in_roi = get_xy_coords_in_ROIs(width, height, ROI_dict)
    
    # loop through each dpf and fit a KDE for observations on that dpf
    dpfs = np.sort(dataframe['dpf'].unique())
    
    os.makedirs(kde_folder_path, exist_ok=True)
    start = time()
    min_kde_val = 1.0
    max_kde_val = 0.0
    for dpf in dpfs:
        print(f'Fitting model for dpf: {dpf}', flush=True)
        dpf_df = dataframe[dataframe['dpf'] == dpf].copy()
        dpf_df['xcenter_normalized'] *= width
        dpf_df['ycenter_normalized'] *= height
        
        model = gaussian_kde(dpf_df[['xcenter_normalized', 'ycenter_normalized']].T, bw_method=density_bandwidth)
        values = model(xy_coords_in_roi.T)
        max_kde_val = max(max_kde_val, np.max(values))
        min_kde_val = min(min_kde_val, np.min(values))
        
        current_kde_values_path = os.path.join(kde_folder_path, f'kde_values_dpf_{dpf}_bandwidth_{density_bandwidth}.txt')
        np.savetxt(current_kde_values_path, values)
    end = time()
    print('Min:', min_kde_val, 'Max:', max_kde_val, flush=True)
    print('Log Min:', np.log10(min_kde_val), 'Log Max:', np.log10(max_kde_val), flush=True)
    print('Time taken: {:.2f} seconds'.format(end-start), flush=True)
    return

def plot_kde_dpf_plot(dataframe, width, height, ROI_dict, backgd_img, kde_folder_path, bandwidth, vmin, vmax, crop_xmin, crop_xmax, crop_ymin, crop_ymax, alpha=0.5, log_scale=False):
    # get a list of the (x,y) coordinates which are in the ROIs.
    xy_coords_in_roi = get_xy_coords_in_ROIs(width, height, ROI_dict)
    xs, ys = xy_coords_in_roi.T
    
    # create a 3-channel backgd image
    backgd_img_rgb = np.zeros((backgd_img.shape) + (3,), dtype=float)
    backgd_img_rgb[...,0] = backgd_img_rgb[...,1] = backgd_img_rgb[...,2] = backgd_img
    
    # choose KDE colormap
    normalize = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('inferno')
    kde_values_paths = glob(os.path.join(kde_folder_path, f'*bandwidth_{bandwidth}.txt'))
    print(kde_values_paths)
    # loop through each dpf and load saved dpf-specific KDE values
    dpfs = np.sort(dataframe['dpf'].unique())
    print('Total dpfs:', dpfs)
    num_cols = 4
    fig, axs = plt.subplots(figsize=(13, 10), nrows=len(dpfs)//num_cols, ncols=num_cols)
    min_kde_val = 1.0
    max_kde_val = 0.0
    for dpf in dpfs:
        dpf_img = backgd_img_rgb.copy()
        current_dpf_kde_values_path = [x for x in kde_values_paths if f'dpf_{dpf}' in x][0]
        kde_values = np.loadtxt(current_dpf_kde_values_path)
        if log_scale == True:
            kde_values = np.log10(kde_values)
        max_kde_val = max(max_kde_val, np.max(kde_values))
        min_kde_val = min(min_kde_val, np.min(kde_values))
        kde_values_colors = cmap(normalize(kde_values), bytes=True)[...,:3].astype(float)
        
        dpf_img[(ys, xs)] *= (1-alpha)
        dpf_img[(ys, xs)] += alpha*kde_values_colors
        
        axs[dpf//num_cols][dpf%num_cols].imshow(dpf_img.astype(np.uint8)[crop_ymin:crop_ymax, crop_xmin:crop_xmax])
        axs[dpf//num_cols][dpf%num_cols].set_xticks([])
        axs[dpf//num_cols][dpf%num_cols].set_yticks([])
        axs[dpf//num_cols][dpf%num_cols].text(100, 150, f'dpf: {dpf}', fontsize='small')
    print('Min:', min_kde_val, 'Max:', max_kde_val, flush=True)
    return fig