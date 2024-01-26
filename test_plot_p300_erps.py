# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:05:42 2024

File: test_plot_p300_erps.py
Authors: Nicholas Kent
Date: 1/25/2024
Description:
"""
#%%
# Part 1
# Import previous module
from load_p300_data import load_training_eeg

# Define subject variable
subject_3 = 3

# Define data directory variable
data_directory = "./P300Data"

# Call the previous module
eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject_3, data_directory)

#%%
# Part 2
# Import get_events method
from plot_p300_erps import get_events

# get_events call
event_sample, is_target_event = get_events(rowcol_id, is_target)

