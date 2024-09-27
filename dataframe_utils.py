import numpy as np
import matplotlib.pyplot as plt # plotting library
import pandas as pd # read and tidy up data
import os
import glob
import json # useful for loading json files (such as the ROI file)
import datetime as dt # useful for doing datetime arithmetic (subtracting days from each other)
from datetime import timedelta 
from matplotlib.path import Path # the following library contains useful polygon representations that allow us to easily check wether a point is inside the polygon (ROI) or not


# called in add_time_col function
def extract_date_time_old(x):
    x = x[:-5]
    date_time= dt.datetime.strptime(x, "%y%m%d%H%M%S")
    return date_time

def extract_date_time(x):
    # find position of first underscore
    idx1 = x.index("_")
    date = x[:idx1]
    
    # find position of second underscore
    x = x[idx1+1:]
    idx2 = x.index("_")
    time = x[:idx2]
    
    date_time= dt.datetime.strptime(date+"-"+time, "%Y%m%d-%H%M%S")
    return date_time


def add_time_col(dataframe):
    if '_' in dataframe["file_name"].iloc[0]:
        contains_underscore = True
    else:
        contains_underscore = False
        
    if contains_underscore:
        dataframe["time_of_day"] = dataframe["file_name"].apply(extract_date_time)
    else:
        dataframe["time_of_day"] = dataframe["file_name"].apply(extract_date_time_old)
    
    dataframe = dataframe.sort_values(by=["time_of_day"])
    return dataframe

# Creat a column of absolute time to use in cumulative plots
def extract_relative_time(x, start_frame):
    relative_time = x - start_frame
    return relative_time.total_seconds

def add_relative_time_col(dataframe):
    start_frame = dataframe["time_of_day"].min()
    dataframe["relative_time"] = dataframe["time_of_day"].apply(lambda x: extract_relative_time(x, start_frame)) # lambda is an anonymous function. We need it to be able to pass 2 parameters
    return dataframe

# Extract dpf by subtracting the start timestamp from each timestamp
def extract_dpf(x, start_frame):
    dpf = x - start_frame
    return dpf.days
    
# Extract dpf from earliest time_of_day (OLD)
# Could use a for loop but slow in python, so we use the apply function, which applies a function to each element in a series (column)
def add_dpf_col_fish_time(dataframe):
    start_frame = dataframe["time_of_day"].min()
    dataframe["dpf"] = dataframe["time_of_day"].apply(lambda x: extract_dpf(x, start_frame)) # lambda is an anonymous function. We need it to be able to pass 2 parameters
    return dataframe

# Extract dpf from earliest time_of_day (NEW)
def add_dpf_col_human_time(dataframe):
    start_dpf = dataframe["time_of_day"].min()
    start_dpf = start_dpf.replace(hour=0, minute=0, second=0)
    dataframe["dpf"] = dataframe["time_of_day"].apply(lambda x: extract_dpf(x, start_dpf)) # lambda is an anonymous function. We need it to be able to pass 2 parameters
    return dataframe

def add_dayslice_col(dataframe):
    dataframe["day_slice"] = dataframe.groupby("dpf")["time_of_day"].rank("dense", ascending=True).astype(int)
    return dataframe

# https://matplotlib.org/stable/tutorials/advanced/path_tutorial.html
def get_ROI_dict(ROI_file_path):
    with open(ROI_file_path) as coords: # open ROI json path
        ROI_json = json.load(coords)
    
    ROI_coords = {} # create dictionary  with key-value stores
    for i, roi in enumerate(ROI_json["shapes"]):
        draw_codes = [Path.MOVETO] + [Path.LINETO] * (len(roi['points']) - 2) + [Path.CLOSEPOLY]
        ROI_coords[roi['label']] = Path(roi['points'], draw_codes)
    return ROI_coords

def extract_ROI(xpos, ypos, width, height, ROI_coords):
    xpos_unnormalized = xpos * width
    ypos_unnormalized = ypos * height
    for roi_label, roi_polygon in ROI_coords.items():
        if roi_polygon.contains_point((xpos_unnormalized, ypos_unnormalized)):
            return roi_label
    return 'outside' # incase the point is not inside any ROI, we count it to be outside

def add_ROI_col(dataframe, width, height, ROI_coords):
    # apply the above function to the x and y coordinates in each row of our larvae dataframe
    dataframe["ROI"] = dataframe.apply(lambda row: extract_ROI(row['xcenter_normalized'], row['ycenter_normalized'], width, height, ROI_coords), axis=1)
    return dataframe

def split_df_at_time(df, time_of_split_str):
    time_of_split = dt.datetime.strptime(time_of_split_str, "%Y%m%d_%H%M%S")
    df_before = df[df['time_of_day'] <= time_of_split]
    df_after = df[df['time_of_day'] > time_of_split]
    return df_before, df_after 

#This create the bin sizes for the light-dark plot
def assign_day_period(df, num_periods):
       
    # Extract the hour and minute components from the datetime
    df['hour'] = df['time_of_day'].dt.hour
    df['minute'] = df['time_of_day'].dt.minute
    
    # Calculate the period length in minutes
    period_length = 24 * 60 // num_periods
    
    # Calculate the minute offset for each period
    period_offset = (df['hour'] * 60 + df['minute']) // period_length
    
    # Create the "day_period" column
    df['day_period'] = period_offset + 1
    
    # Drop the temporary columns
    df = df.drop(columns=['hour', 'minute'])
    
    return df
