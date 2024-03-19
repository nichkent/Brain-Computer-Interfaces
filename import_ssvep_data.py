# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:33:12 2024

File: import_ssvep_data.py
Authors: Nicholas Kent, 
Date: 2/23/2024
Description: This script, import_ssvep_data.py, loads data of two subjects from an SSVEP experiment with the function load_ssvep_data.
The script then plots the data on a graph for better visualization with the function plot_raw_data. Next, the function epoch_ssvep_data 
iterates over events in an EEG dataset, extracting segments (epochs) of data starting from a specified start time to an end time relative 
to each event's onset. It constructs an array of these epochs, an array of corresponding times, and a boolean array indicating whether the 
event is associated with a 15Hz flickering stimulus. The Forier Transform of the EEG epochs are then taken with a sampling frequency supplied
from the data with the function get_frequency_spectrum. Finally, the function plot_power_spectrum plots the Forier Transform on a graph for easier
visualization of the data.
"""
#%%
# Part 1
# Imports
import numpy as np
from matplotlib import pyplot as plt
import scipy.fft

def load_ssvep_data(subject="1", data_directory='./SsvepData'):
    """
    Loads the raw data from a specific subjects .npz file.
    
    Parameters
    ----------
    subject : str, optional
        Holds the number that maps to a subject in the dataset. The default is "1".
    data_directory : str, optional
        The directory where the SSVEP data is stored. The default is './SsvepData'.

    Returns
    -------
    data_dict : dict
        Dictionary containing the EEG data and metadata. The structure is as follows:
        - 'eeg': 2D array (channels x samples) containing EEG data in volts.
        - 'channels': 1D array listing the names of EEG channels.
        - 'fs': Sampling frequency in Hz (scalar).
        - 'event_samples': 1D array indicating the samples where each event occurred.
        - 'event_durations': 1D array with the durations of each event in samples.
        - 'event_types': 1D array listing the frequency of the flickering checkerboard for each event.

    Note: The exact size of the 'eeg' array and the length of the 'event_samples', 'event_durations', and 'event_types' arrays
    depend on the specific subject and the number of recorded events.

    """
    
    # Load the data from the given subject's file using the data directory
    data_from_file = np.load(f'{data_directory}/SSVEP_S{subject}.npz', allow_pickle=True)
    
    # Extract data components
    eeg = data_from_file['eeg']  # The EEG data in volts
    channels = data_from_file['channels']  # The names of each channel
    fs = data_from_file['fs']  # Sampling frequency
    event_samples = data_from_file['event_samples']  # The sample when each event occurred
    event_durations = data_from_file['event_durations']  # The durations of each event in samples
    event_types = data_from_file['event_types']  # The frequency of flickering checkerboard for each event
    
    # Organize data into a dictionary
    data_dict = {
        'eeg': eeg,
        'channels': channels,
        'fs': fs,
        'event_samples': event_samples,
        'event_durations': event_durations,
        'event_types': event_types
    }
    
    # Return the fully formed data dictionary with all data from the given subject
    return data_dict

#%%
# Part 2
def plot_raw_data(data, subject, channels_to_plot):
    """
    Plots the raw EEG data and event markers for specified channels.

    Parameters
    ----------
    data : dict
        The EEG data dictionary loaded from the npz file.
    subject : int or str
        The subject number.
    channels_to_plot : list of str
        The list of channel names to plot.
        
    Returns
    ---------
    None (Creates a plot for the given subjet's raw EEG data and event markers for espcified channels)
    
    """
    
    # Extract data components
    eeg = data['eeg']  # The EEG data in volts
    channels = data['channels']  # The names of each channel
    fs = data['fs']  # Sampling frequency
    event_samples = data['event_samples']  # The sample when each event occurred
    event_durations = data['event_durations']  # The durations of each event in samples
    event_types = data['event_types']  # The frequency of flickering checkerboard for each event

    # Convert sample indices to time in seconds
    time_in_sec = np.arange(eeg.shape[1]) / fs

    # Create the figure and two subplots (ax_event and ax_eeg) with shared a x-axis (sharex=True)
    fig, (ax_event, ax_eeg) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # Plot events in the first subplot
    # Use python's zip function to align each event_sample, event_dureation, and event_type one to one
    for start_sample, duration_samples, event_type in zip(event_samples, event_durations, event_types):
        
        # Calculate the start and end times of the event in seconds using the sampling frequency
        start_time = start_sample / fs
        end_time = (start_sample + duration_samples) / fs
        
        # Check if event_type is a string and convert to float, removing 'hz' if necessary
        event_type_num = float(event_type.replace('hz', ''))

        # Plot the event with a start and end marker on the same y-value
        ax_event.plot([start_time, end_time], [event_type_num, event_type_num], marker='o', markersize=4, linestyle='-', color='blue')
            
    # Set labels and limits for the event plot
    ax_event.set_ylim([11.9, 15.1])
    ax_event.set_ylabel('Flash Frequency (Hz)')
    ax_event.set_title(f'SSVEP subject {subject} Raw Data')

    # Plot the raw EEG data for the specified channels in the second subplot
    # Plotting data according to where the channel name in chanels_to_plot matches the channel name in channels
    channel_indices = [np.where(channels == current_channel)[0][0] for current_channel in channels_to_plot]
    
    # Iterate through channel_indices to find the x-axis for the plot multiplying by eeg[current_index, :] by 1e6 to match the sample graph
    for current_index in channel_indices:
        ax_eeg.plot(time_in_sec, eeg[current_index, :] * 1e6, label=channels[current_index])

    # Set labels and titles for the eeg voltage plot
    ax_eeg.set_xlabel('Time (s)')
    ax_eeg.set_ylabel('Voltage (uV)')    
    ax_eeg.legend()

    # Adjust the plot to tighten layout
    plt.tight_layout()
    
    # Add gridlines to the first plot
    ax_event.grid()
    
    # Add gridlines to the second plot
    ax_eeg.grid()

    # Save the figure
    plt.savefig(f'SSVEP_{subject}_rawdata.png')

    # Show the plot
    plt.show()
    
#%%
# Part 3
def epoch_ssvep_data(data_dict, epoch_start_time="0", epoch_end_time="20"):
    """
    Extract epochs from EEG data at specified intervals relative to events.

    This function iterates over events in an EEG dataset, extracting segments (epochs) of data starting from 
    a specified start time to an end time relative to each event's onset. It constructs an array of these 
    epochs, an array of corresponding times, and a boolean array indicating whether the event is associated 
    with a 15Hz flickering stimulus.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing the EEG data and metadata. The structure is as follows:
        - 'eeg': 2D NumPy array with shape (samples, channels) containing EEG data in volts.
        - 'fs': Sampling frequency in Hz (scalar).
        - 'event_samples': 1D array indicating the samples where each event occurred.
        - 'event_types': 1D array listing the frequency of the flickering checkerboard for each event.
    epoch_start_time : int or float, optional
        The start time in seconds for each epoch, relative to the event onset. The default is 0.
    epoch_end_time : int or float, optional
        The end time in seconds for each epoch, relative to the event onset. The default is 20.

    Returns
    -------
    eeg_epochs : 3D NumPy array
        An array of EEG epochs with dimensions (trials, channels, time).
    epoch_times : 1D NumPy array
        An array of times in seconds for each point in the EEG epochs array, relative to the event onset.
    is_trial_15Hz : 1D NumPy array of bool
        A boolean array indicating whether the light was flickering at 15Hz during each trial/epoch.
    
    """
    
    # Extract data components
    eeg = data_dict['eeg']
    fs = data_dict['fs']
    event_samples = data_dict['event_samples']
    event_types = data_dict['event_types']
    
    # Calculate the number of samples to offset for start and end times
    start_offset_samples = int(epoch_start_time * fs)
    end_offset_samples = int(epoch_end_time * fs)

    # Initialize an empty list to store epochs
    eeg_epochs = []
    is_trial_15Hz = []

    # Iterate over each event in event_samples
    for event_sample, event_type in zip(event_samples, event_types):
        # Calculate the start and end sample indices for the epoch
        start_sample = event_sample + start_offset_samples
        end_sample = event_sample + end_offset_samples

        # Check if the epoch is within the bounds of the EEG data from data_dict
        if 0 <= start_sample < eeg.shape[1] and 0 < end_sample <= eeg.shape[1]:
            # Extract the epoch with the calculated start and end samples
            current_epoch = eeg[:, start_sample:end_sample]

            # Store the epoch data in an array eeg_epochs
            eeg_epochs.append(current_epoch)

            # Check if this trial is a 15Hz trial and store the result in an array is_trail_15Hz
            if event_type == '15hz':
                is_trial_15Hz.append(True)
            else:
                is_trial_15Hz.append(False)

    # Check to see if the epochs were extracted correctly
    if eeg_epochs:
        # Change eeg_epochs into a numpy array for ease of use later
        eeg_epochs = np.array(eeg_epochs)
        
        # Change eeg_epochs into a numpy array for ease of use later
        is_trial_15Hz = np.array(is_trial_15Hz)
    else:
        # Handle the case where no epochs were extracted, set a default for each case empty state
        eeg_epochs = np.empty((0, eeg.shape[0], end_offset_samples - start_offset_samples))
        is_trial_15Hz = np.array([])

    # Generate the epoch times array for later use
    epoch_times = np.linspace(epoch_start_time, epoch_end_time, end_offset_samples - start_offset_samples)

    # Return epochs as a 3D array, epoch times for later use, and trials at 15Hz
    return eeg_epochs, epoch_times, is_trial_15Hz

#%%
# Part 4

def get_frequency_spectrum(eeg_epochs, fs):
    """
    This function calculates the Fourier Transform of EEG data with a specific sampling freqeuncy accross a 3D array.

    Parameters
    ----------
    eeg_epochs : 3D NumPy array
        An array of EEG epochs with dimensions (trials, channels, time).
    fs : float
        Sampling freqeuncy in Hz.

    Returns
    -------
    eeg_epochs_fft : 3D NumPy array
        An array of EEG epochs with dimensions (trials, channels, frequencies).
    fft_frequencies : 1D NumPy array
        An array of frequencies corresponding to eeg_epochs_fft, length: eeg_epochs_fft.shape[2].

    """
    
    
    # T (period) is the total duration divided by fs
    T = eeg_epochs.shape[2]/fs
    
    # calculate the fft of each channel in each trial 
    eeg_epochs_fft = scipy.fft.rfft(eeg_epochs, axis = 2)

    # establish frequency array
    fft_frequencies = np.arange(0,(fs/2)+(1/T),1/T)
    
    return eeg_epochs_fft, fft_frequencies


#%%

# Part 5

def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject):
    """
    
    This function performs calculations necessary to plot the frequency spectrums of specified channels of one 
    subjects EEG data, split into 15 and 12 Hz trials.

    Parameters
    ----------
    eeg_epochs_fft : 3D NumPy array
        An array of EEG epochs with dimensions (trials, channels, frequencies).
    fft_frequencies : 1D NumPy array
        An array of frequencies corresponding to eeg_epochs_fft, length: eeg_epochs_fft.shape[2].
    is_trial_15Hz : 1D NumPy array of bool
        A boolean array indicating whether the light was flickering at 15Hz during each trial/epoch.
    channels : Array of str
        array containing the names of EEG electrodes included in the dataset.
    channels_to_plot : list of str
        The list of channel names to plot.
    subject : int
        The subject number.

    Returns
    -------
    spectrum_db_12Hz : 2D NumPy array 
        An array contianing the mean power spectrum of 12Hz trials in dB for all trials (number of channels, freqeuncies).
    spectrum_db_15Hz : 2D NumPy array
        An array contianing the mean power spectrum of 15Hz trials in dB for all trials (number of channels, freqeuncies).

    """
    
    # split into 15 and 12 Hz epochs
    trials_12Hz = eeg_epochs_fft[is_trial_15Hz,:,:]
    trials_15Hz = eeg_epochs_fft[~is_trial_15Hz,:,:]
    
    # calculate power
    power_12Hz = np.square(abs(trials_12Hz))
    power_15Hz = np.square(abs(trials_15Hz))
    
    # take mean power across trials
    mean_power_12Hz = np.mean(power_12Hz, axis = 0)
    mean_power_15Hz = np.mean(power_15Hz, axis = 0)
    
    # normalize power
    norm_power_12Hz = mean_power_12Hz/np.max(mean_power_12Hz)
    norm_power_15Hz = mean_power_12Hz/np.max(mean_power_15Hz)
    
    # convert to dB
    spectrum_db_12Hz = 10*np.log10(norm_power_12Hz)
    spectrum_db_15Hz = 10*np.log10(norm_power_15Hz)
    
    # extract channel indices
    channel_indices = [np.where(channels == current_channel)[0][0] for current_channel in channels_to_plot]
    
    plt.figure()
    
    for channel_index, channel in enumerate(channel_indices):
        channel_name = channels_to_plot[channel_index]
        
        plt.subplot(2,1,channel_index+1)
        plt.plot(fft_frequencies, spectrum_db_12Hz[channel,:], 'red', label = '12 Hz')
        plt.plot(fft_frequencies, spectrum_db_15Hz[channel,:], 'green', label = '15 Hz')
        plt.axvline(x=12, color='red', linestyle=':')
        plt.axvline(x=15, color='green', linestyle=':')
        plt.title(f'Channel {channel_name} freqeuncy content for SSVEP S{subject}')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('power (dB)')
        plt.xlim(0,80)
        plt.legend()
        plt.tight_layout()
        
    return spectrum_db_12Hz, spectrum_db_15Hz
  
    
    
    
    
    
