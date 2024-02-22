#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file's purpose is processing and analyzing EEG data.

Part 1: Setup
  Import necessary Python packages

Part 2: Event Detection
  Define a function to determine the start of stimulus events; categorize them as target/non-target events.

Part 3: Exploratory Data Analysis
  Functions to establish sampling rate and to segment EEG data into epochs.
  Constants defining sample rate and channel count.

Part 4: ERP Extraction
    Extract the event-related potentials (ERPs) for target and non-target events.

Part 5: Visualization
  Plotting functions for the visualization of ERP averages across EEG channels.

@author: Aiden Pricer-Coan
@author: Michael Gallo

file: plot_p300_erps.py
BME 6710 - Jangraw
Lab Two: Event-Related Potentials
"""

# Import statements
#%% Part 1
import numpy as np
import matplotlib.pyplot as plt

#%% Part 2

def get_events(rowcol_id, is_target):
    """
        Loads training data from filename calculated with the subject number and 
        given data_directory.
        
        Args:
         - rowcol_id <np.array (int)>[TOTAL_SAMPLES] : array of integers representing the flashed row/column identifier being flashed for each datapoint.
         - is_target <np.array (bool)>[TOTAL_SAMPLES] : array of booleans representing whether or not the flashed row/column contains the target letter for each datapoint
        
        Returns:
         - event_sample <np.array (int)>[TOTAL_SAMPLES] : indices where every new event begins.
         - is_target_event <np.array (bool)>[TOTAL_SAMPLES] : array where each element indicates whether an event was a target event or not.
    """
    # Calculate the difference between rowcol_id vals
    diff_rowcol_id = np.diff(rowcol_id)
    
    # Find each index before where the value increases from 0
    #   Add 1, because we don't need the value BEFORE change...
    event_sample = np.where(diff_rowcol_id > 0)[0] + 1
    
    # Create array to denote whether is_target for each event
    is_target_event = is_target[event_sample]
    
    # Return the indices of event start and whether those events are target events
    return event_sample, is_target_event

#%% Part 3 - Exploratory Data Analysis:
    
def num_samples_in_sec(eeg_time):
    """
        Used to find the number of samples per second of data.
        
        Args:
         - eeg_time <np.array (float)>[TOTAL_SAMPLES] : array of floats representing the time value in seconds at each index.
        
        Returns:
         - time_index <int> : number of elements within 1s of data
    """
    first_time = eeg_time[0]
    time_index = 1
    while eeg_time[time_index] - first_time != 1:
        print("time passed", eeg_time[time_index] - first_time)
        time_index += 1
    print(f"Number of elements within 1s of data: {time_index}")
    return time_index

#%% Part 3

# Calculated above AND confirmed in documentation
SAMPLES_PER_SECOND = 256
NUM_CHANNELS = 8

def epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time=-0.5, epoch_end_time=1):
    """
        Loads our data into epoch blocks for further analysis.
        
        Args:
         - eeg_time <np.array>[TOTAL_SAMPLES] : Array of floats representing the time value in seconds at each index.
         - eeg_data <np.array>[TOTAL_SAMPLES, NUM_CHANNELS] : 2d array with EEG values for each channel at every data point in the experiment data.
             - NOTE: eeg_data[i, j], where i represents the i(th) channel
                                           j represents the j(th) element
         - event_sample <np.array>[NUM_EPOCHS] : Integer indices where every new event begins.
         - epoch_start_time <float> : start time offset from start point of each epoch, can be + or -
         - epoch_end_time <float> : end time offset from start point of each epoch, should be > epoch_start_time
    
        Returns:
         - eeg_epochs <np.array>[NUM_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS]: 3d array representing an epoch for each event in our data
             - eeg_epochs[i][j][k], where i represents the i(th) epoch, 
                                          j represents the j(th) sample in the epoch,
                                          k represents the k(th) channel of data in the epoch.
         - erp_times <np.array>[SAMPLES_PER_EPOCH]: 1d array of floats representing the time difference of each datapoint from the start of the epoch's event
    """
    # Calculate # of seconds in a single epoch
    seconds_per_epoch = epoch_end_time - epoch_start_time
    # Calculate # of samples in a single epoch
    samples_per_epoch = round(SAMPLES_PER_SECOND * seconds_per_epoch)
    # Number of epochs...
    num_epochs = len(event_sample)
    
    # Create a 3D array of zeros with correct shape
    eeg_epochs = np.zeros([num_epochs, samples_per_epoch, NUM_CHANNELS])
    
    # Get times at which each event starts...
    event_start_times = eeg_time[event_sample]
    
    # Enumerate through each event, creating an epoch with extracted data
    for event_number, event_start_time in enumerate(event_start_times):
        # Define the epoch window start and end times
        window_start_time = event_start_time + epoch_start_time
        window_end_time = event_start_time + epoch_end_time
        
        # Get indices within window...
        window_indices = np.where((eeg_time >= window_start_time) & (eeg_time < window_end_time))[0]
        # Get epoch data, transpose because (eeg_data[i][j])'s i represents channel, but we NEED i to represent the sample index, and j the channel
        epoch_data = eeg_data[:, window_indices].T
        # Set the epoch data
        eeg_epochs[event_number, :, :] = epoch_data
    
    # Create erp_times array
    time_step = 1 / SAMPLES_PER_SECOND
    erp_times = np.arange(epoch_start_time, epoch_end_time, time_step)

    return eeg_epochs, erp_times

#%% Part 4

def get_erps(eeg_epochs, is_target_event):
    """
        Extract the event-related potentials (ERPs) for target and non-target events.
    
        Args:
         - eeg_epochs <np.array>[NUM_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS]: 3d array representing an epoch for each event in our data
             ~ eeg_epochs[i][j][k], where i represents the i(th) epoch, 
                                          j represents the j(th) sample in the epoch,
                                          k represents the k(th) channel of data in the epoch.
         - is_target_event <np.array (bool)>[NUM_EPOCHS]: Array of booleans indicating whether each event is a target event or not
        
        Returns:
         - target_erp <np.array>[NUM_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : ERP floats for target events
         - nontarget_erp <np.array>[NUM_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : ERP floats for non-target events
    """
    # Select epochs corresponding to target events
    target_epochs = eeg_epochs[is_target_event]
    # Select epochs corresponding to non-target events
    nontarget_epochs = eeg_epochs[~is_target_event]

    # Calculate mean response on each channel for target events
    target_erp = np.mean(target_epochs, axis=0)
    # Calculate mean response on each channel for non-target events
    nontarget_erp = np.mean(nontarget_epochs, axis=0)

    return target_erp, nontarget_erp

#%% Part 5

def plot_erps(target_erp, nontarget_erp, erp_times):
    """
        Plots event-related potentials (ERPs) for target and non-target events.
    
        Args:
         - target_erp <np.array>[NUM_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : Array containing the mean response on each channel for target events.
         - nontarget_erp <np.array>[NUM_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : Array containing the mean response on each channel for non-target events.
         - erp_times (np.array)[SAMPLES_PER_EPOCH]: Array representing the time difference of each datapoint from the start of the event.
    
        Returns:
         - None
    """
    # Create figure and subplots
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    axes = axes.flatten()

    # Iterate through the axes and plot target/non-target data
    for channel_number in range(NUM_CHANNELS):
        # Plot target ERP for this channel
        axes[channel_number].plot(erp_times, target_erp[:, channel_number], label='Target')
        
        # Plot nontarget ERP for this channel
        axes[channel_number].plot(erp_times, nontarget_erp[:, channel_number], label='Non-target')
        
        # Add reference lines at x=0 and y=0
        axes[channel_number].axhline(0, color='black', linestyle='dotted')
        axes[channel_number].axvline(0, color='black', linestyle='dotted')
        
        # Set title to the channel name
        axes[channel_number].set_title(f"Channel {channel_number}")
        
        # Label axes
        axes[channel_number].set_xlabel('time from flash onset (s)')
        axes[channel_number].set_ylabel('Voltage (µV)')
        
        # Add legend
        if channel_number == NUM_CHANNELS - 1:
            axes[channel_number].legend()
        
    # Hide last unused plot...
    for unused_axes_index in range(NUM_CHANNELS, len(axes)):
        axes[unused_axes_index].axis('off')
    
    plt.tight_layout()
    plt.show()
