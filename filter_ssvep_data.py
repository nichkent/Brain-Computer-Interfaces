# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:28:07 2024

File: filter_ssvep_data.py
Authors: Nicholas Kent, 
Date: 3/7/2024
Description: This script, filter_ssvep_data.py,
"""

#%%
# Part 2

# Import libraries
import numpy as np
from scipy.signal import firwin, freqz, filtfilt
import matplotlib.pyplot as plt

def make_bandpass_filter(low_cutoff, high_cutoff, filter_type="hann", filter_order=10, fs="1000"):
    """
    Creates a bandpass filter and plots its impulse and frequency responses.
    
    Parameters
    ----------
    low_cutoff : int
        The low cutoff frequency of the bandpass filter in Hz.
    high_cutoff : int
        The high cutoff frequency of the bandpass filter in Hz.
    filter_type : str
        The window type to use in the design of the FIR filter. Common types include 'hamming', 'hann', and 'blackman'.
    filter_order : int
        The order of the filter. A higher order will result in a steeper roll-off.
    fs : int
        The sampling frequency of the signal to be filtered in Hz.
    
    Returns
    -------
    filter_coefficients : 1d-array
        The coefficients of the designed bandpass FIR filter. Length of filter_order + 1.
    
    Notes
    -----
    The method uses the `firwin` function from `scipy.signal` to create the filter coefficients based on the specified parameters. 
    It then calculates the frequency response of the filter and plots both the impulse response and frequency response. 
    The impulse response is plotted as a line plot for better visualization, and the frequency response is plotted in decibels (dB) 
    against frequency in Hz. The plot includes titles, axis labels, and a grid for clarity. Additionally, the plot is saved as a PNG file 
    with a filename indicating the filter's cutoff frequencies and order.
    
    The specific characteristics of the filter (such as the roll-off, stopband attenuation, and passband ripple) depend on the chosen `filter_type` 
    and `filter_order`. Users should choose these parameters based on the requirements of their application.
    
    """   
    # Create a bandpass filter using the firwin function from scipy.signal
    # Uses the low and high cutoff to create a bandpass filter
    filter_coefficients = firwin(filter_order + 1, [low_cutoff, high_cutoff], pass_zero=False, window=filter_type, fs=fs)
    
    # Calculate the frequency response of the filter
    freq_response, response = freqz(filter_coefficients, worN=2048, fs=fs)  # Increase worN for better frequency resolution, 2048 because it's a number divisible by 2
    
    # Plot the impulse response and frequency response
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    
    # Set up for the impulse response graph
    # Make a time numpy array from 0 to the filter_order + 1.
    time = np.arange(0, filter_order + 1) / fs   
    
    axs[0].plot(time, filter_coefficients, 'dodgerblue')
    axs[0].set_title('impulse response')
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('gain')
    axs[0].set_xlim(-0.1, 1.1)
    axs[0].set_ylim(-0.006, 0.006)
    axs[0].grid()
    
    # Set up for the frequency response graph
    axs[1].plot(freq_response, 20 * np.log10(abs(response)), 'dodgerblue')
    axs[1].set_title('frequency response')
    axs[1].set_xlabel('frequency (Hz)')
    axs[1].set_ylabel('amplitude (dB)')
    axs[1].set_xlim(0, 40)
    axs[1].set_ylim(-250, 5)
    axs[1].grid()
    
    # Name the graph
    fig.suptitle(f"Bandpass {filter_type} filter with fc=[{low_cutoff}, {high_cutoff}], order={filter_order + 1}")
    
    # Show the graph in a tight layout
    plt.tight_layout()
    
    # Save the plot
    filename = f"filter_response_{low_cutoff}-{high_cutoff}_order{filter_order}.png"
    plt.savefig(filename)
    
    # Show the plot
    plt.show()
    
    # Return the filter coefficients
    return filter_coefficients

#%% Part 3: Filter EEG Signals

def filter_data(data, b):
    """
    Applies the bandpass filter to EEG data for each channel for one subject. 
    The filter is applied forwards and backwards in time to each channel of 
    the raw EEG data using scipy's filtfilt function. 

    Parameters
    ----------
    data : Dict of size 6.
        A dictionary where keys are channels, eeg, event_durations, 
        event_samples, event_types, and fs. Values are arrays, details of each
        array follow. 
        - Channels: Array of str, size (C,) where C is the number of channels.
        Values represent channel names, for example "Cz".
        - eeg: Array of float, size (C,S) where C is the number of channels and
        S is the number of samples. Values represent raw EEG data in volts.
        - event_durations: Array of float, size (E,) where E is the number of 
        events. Values represent the duration of each event in samples.
        - event_samples: Array of int, size (E,) where E is the number of events.
        Values represent the sample when each event occurred.
        - event_types: Array of object, size (E,) where E is the number of events.
        Values represent the frequency of flickering checkboard that started 
        flashing for an event (12hz or 15hz)
        - fs: Array of float, size 1. The sampling frequency in Hz.
    b : Array of float, size (O+1,) where O is the order of the filter. In our
    case, the filter order is 1000 so b is size (1001,)
        The filter coefficients. 

    Returns
    -------
    filtered_data : Array of float, size (C,S) where C is the number of channels 
    present in the raw EEG data and S is the numebr of samples present in the 
    raw EEG data.
        Filtered EEG data in uV.

    """
    # apply filter forwards and backwards in time to each channel in raw data
    # initialize variable to hold filtered data
    filtered_data=np.zeros((data["eeg"].shape))

    # under eeg key, each row is a channel and each column is a sample
    for channel in range(data["eeg"].shape[0]):
        # get filtered data for a channel in v
        filtered_data[channel] = filtfilt(b, 1, data["eeg"][channel])
        filtered_data[channel] = filtered_data[channel] * 1000000 # convert to uv
    return filtered_data
