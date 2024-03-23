# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:28:07 2024

File: filter_ssvep_data.py
Authors: Nicholas Kent, Alaina Birney
Date: 3/7/2024
Description: This script, filter_ssvep_data.py, focuses primarily on isolating and analyzing
Stead-State Visually Evoked Potentials (SSVEP) within specific frequency bands. Firstly, the
script begins by creating bandpass filters, to filter the EEG data to include only the frequencies
of interest (12Hz and 15Hz). This reduces noise and irrelevent frequency components that would otherwise
change analysis of the data. The script then applies these filters to the EEG data, generating filtered signals
that are more representative of the underlying neural response to SSVEP stimulus from the particpant. Then, the script
calculates the envelopes of these signals, highlighting amplitude modulation within each band, and provides
a clear indication of the temporal spectra at different stages (i.e. raw, filtered, and envelope) for each channel.
"""

#%%
# Part 2

# Import libraries
import numpy as np
from scipy.signal import firwin, freqz, filtfilt, hilbert
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
    freq_points, freq_response = freqz(filter_coefficients)
    
    # Calculate frequency for the domain of the frequency graph
    freq = freq_points * fs / (2 * np.pi)
    
    # Plot the impulse response and frequency response as subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    
    # Set up for the impulse response graph
    # Make a time numpy array from 0 to the filter_coefficeints divided by the sampling frequency
    time = np.arange(0, len(filter_coefficients)) / fs
    
    # Use the time as the domain for the impulse graph
    axs[0].plot(time, filter_coefficients, 'dodgerblue')
    axs[0].set_title('impulse response')
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('gain')
    axs[0].set_xlim(-0.1, 1.1)
    axs[0].set_ylim(-0.006, 0.006)
    axs[0].grid()
    
    # Set up for the frequency response graph
    # 10 * np.log10(abs(freq_response)) required for conversion to decibels for the frequency graph
    axs[1].plot(freq, 10 * np.log10(abs(freq_response)), 'dodgerblue')
    axs[1].set_title('frequency response')
    axs[1].set_xlabel('frequency (Hz)')
    axs[1].set_ylabel('amplitude (dB)')
    axs[1].set_xlim(0, 40)
    axs[1].set_ylim(-150, 5)
    axs[1].grid()
    
    # Name the graph
    fig.suptitle(f"Bandpass {filter_type} filter with fc=[{low_cutoff}, {high_cutoff}], order={filter_order}")
    
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
    num_channels = data["eeg"].shape[0]
    
    # loop through channels to filter for each channel and add to filtered_data
    # at channel index
    for channel in range(num_channels):
        # get filtered data for a channel in v
        filtered_data[channel] = filtfilt(b, 1, data["eeg"][channel])
        filtered_data[channel] = filtered_data[channel] * 1000000 # convert to uv
    return filtered_data

#%% Part 4: Calculate the Envelope

def get_envelope(data, filtered_data, channel_to_plot, ssvep_frequency="None"):
    """
    Extract the envelope surrounding a wave of EEG data for every channel 
    at every time point. The envelope connects the peaks in the wave, thus 
    reflecting the wave's amplitude. A plot will be produced for the band-pass 
    filtered EEG data for the channel specified by channel_to_plot with the 
    envelope on top.

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
    filtered_data : Array of float, size (C,S) where C is the number of channels 
    present in the raw EEG data and S is the numebr of samples present in the 
    raw EEG data.
        Filtered EEG data in uV.
    channel_to_plot : Str.
        The channel a user would like to plot the envelope for.
    ssvep_frequency : Str, optional
        The SSVEP frequency being isolated. This is used for the plot title.
        The default is "None".

    Returns
    -------
    envelope : Array of float, size (C,S) where C is the number of channels 
    present in the filtered data and S is the number of samples present in the 
    filtered data.
        The amplitude of oscillations on every channel at every time point.

    """
    # calculate for each channel for each time point
    num_channels = filtered_data.shape[0] # rows are channels
    
    # initialize variable to hold envelope data
    envelope = np.zeros(filtered_data.shape)
    
    # loop through channels to calculate envelope for each channel at each time point
    for channel in range(num_channels):
        envelope[channel] = np.abs(hilbert(filtered_data[channel,:]))
        
    # plot band-pass filtered data on given channel with its envelope on top
    # for channel_to_plot
    # get channel as number from string
    # loop through channels key in data dict, grab index of matching string
    # initialize channel_idx_to_plot in case no match is found
    channel_idx_to_plot = None
    for channel_index, channel_value in enumerate(data["channels"]):
        if channel_value == channel_to_plot:
            channel_idx_to_plot = channel_index
            
    if channel_idx_to_plot is not None:
        plt.figure()
        # plot data from channel of interest
        data_to_plot = filtered_data[channel_idx_to_plot,:]
        # time should be length of total recording duration, spaced by sampling interval
        num_samples = data_to_plot.shape[0]
        fs = data["fs"]
        time = np.arange(0,num_samples)/fs
        plt.plot(time, data_to_plot, label="Band-pass Filtered EEG Data")
        plt.plot(time, envelope[channel_idx_to_plot,:], label="Envelope", color="orange")
        
        # if statement to have title state unknown when ssvep_frequency is None
        if ssvep_frequency == None:
            ssvep_frequency = "Unknown"
            
        plt.title(f"Band-pass Filtered EEG Data for Channel {channel_to_plot}. Isolating {ssvep_frequency} Frequency.")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (uV)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        # save plot
        # Save the plot
        filename = f"BPF_Data_Channel_{channel_to_plot}_Isolating_{ssvep_frequency}.png"
        plt.savefig(filename)
    return envelope

#%% Part 5: Plot the Amplitudes



def plot_ssvep_amplitudes(data, envelope_a, envelope_b, channel_to_plot,
                          ssvep_freq_a, ssvep_freq_b, subject):
    """
    Creates two subplots. The top subplot shows the event types (12 Hz or 15 Hz
    flashing) while the bottom subplot shows the 12 Hz and 15 Hz envelopes.
    This visualization can help one to gain a better sense of whether or not
    a subject is responding to an event.
    Please note that this function includes code that was written with Ron 
    Bryant for lab 3. The code has been modified for this lab. 
    Additionally, please note that avg_dif_a and avg_dif_b were added as return
    values to aid in answering the questions for part 5 in the test script.

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
    envelope_a : Array of float, size (C,S) where C is the number of channels
    and S is the number of samples.
        The envelope of oscillations at the first SSVEP frequency.
    envelope_b : Array of float, size (C,S) where C is the number of channels
    and S is the number of samples.
        The envelope of oscillations at the second SSVEP frequency.
    channel_to_plot : Str
        The channel for which data will be plotted.
    ssvep_freq_a : Str
        The SSVEP frequency being isolated in the first envelope. This is used 
        for the legend only.
    ssvep_freq_b : Str
        The SSVEP frequency being isolated in the second envelope. This is used 
        for the legend only.
    subject : Int
        The subject number for whom data will be plotted. This should correspond
        with the subject number indicated when data was initially loaded.

    Returns
    -------
    avg_dif_a : float
        The average difference (uV) between the two envelopes (envelope_a and
        envelope_b) during ssvep_freq_a events. This metric aims to quantify 
        the relative strength of the EEG response to ssvep_freq_a events
        compared to the response to ssvep_freq_b events by averaging the differences
        in envelope amplitudes across the ssvep_freq_a events. A positive value
        suggests that on average, envelope_a had higher amplitude during the 
        ssvep_freq_a events while a negative value suggests that envelope_b 
        had a higher amplitude during the ssvep_freq_a events.
    avg_dif_b : float
        The average difference (uV) between the two envelopes (envelope_a and
        envelope_b) during ssvep_freq_b events. This metric aims to quantify 
        the relative strength of the EEG response to ssvep_freq_a events
        compared to the response to ssvep_freq_b events by averaging the differences
        in envelope amplitudes across the ssvep_freq_b events. A positive value
        suggests that on average, envelope_a had higher amplitude during the 
        ssvep_freq_b events while a negative value suggests that envelope_b 
        had a higher amplitude during the ssvep_freq_b events.
    """
    #unpack data_dict
    eeg_data = data['eeg']/1e-6   # convert to microvolts
    channels = data['channels']   # channel names
    fs = data['fs']                # sampling frequency
    event_times = data['event_samples']/fs          # convert to seconds
    event_types = data['event_types']        # frequency of event
    event_durations = data['event_durations']/fs    # convert to seconds
    T = eeg_data.shape[1]/fs # Total time
    t = np.arange(0,T,1/fs) # time axis
    
    # initialize figure
    plt.figure(1, clear=True)    
    
    # top subplot: events as horizontal lines with dots at start and end time 
    # of each event (event start and end times)
    ax1 = plt.subplot(211)
    for event_index in range(0,len(event_times)):
        event_time = [event_times[event_index],  
                      event_times[event_index]+event_durations[event_index]]
        event_frequency = [event_types[event_index],event_types[event_index]]
        plt.plot(event_time, event_frequency, 'b.-')
    plt.grid(True)
    plt.ylabel('Flash Frequency')
    plt.xlabel('time (s)')
    
    # second subplot: envelopes of the two filtered signals for subject and 
    # channel to plot
    plt.subplot(212, sharex=ax1) # share x axis
    # get channel number from name to index envelopes
    # initialize channel_idx_to_plot in case no match is found
    channel_idx_to_plot = None
    for channel_index, channel_value in enumerate(channels):
        if channel_value == channel_to_plot:
            channel_idx_to_plot = channel_index
    if channel_idx_to_plot is not None:
        plt.plot(t,envelope_a[channel_idx_to_plot,:], label=f"{ssvep_freq_a} Envelope")
        plt.plot(t,envelope_b[channel_idx_to_plot,:], label=f"{ssvep_freq_b} Envelope")
        plt.grid(True)
        plt.ylabel('Voltage (uV)')
        plt.xlabel('time (s)')
        plt.legend(loc='upper right')
       
        plt.suptitle(f'Subject {subject} SSVEP Amplitudes')
        plt.tight_layout()
        # save to file
        plt.savefig(f"S{subject}_amplitudes.png")
        plt.show()
        
    # the following code was added to aid in answering the questions for part 5
    # find overall difference between envelopes (a-b)
    dif = envelope_a[channel_idx_to_plot, :] - envelope_b[channel_idx_to_plot,:]
    # the event changes at approximately 228 seconds, so sample 228*fs
    change_sample = int(228*fs)
    # get mean difference during SSVEP frequency a events
    avg_dif_a = np.mean(dif[:change_sample])
    # get mean difference during SSVEP frequency b events
    avg_dif_b = np.mean(dif[change_sample+1:])  
    
    return avg_dif_a, avg_dif_b

#%%
# Part 6

# Import
from import_ssvep_data import epoch_ssvep_data, get_frequency_spectrum

def plot_filtered_spectra(data, filtered_data, envelope):
    """
    Plots the power spectra of raw, filtered, and envelope EEG data for specified channels.
    
    This function takes raw EEG data, filtered EEG data, and the envelope of the filtered EEG data, 
    then computes and plots their power spectra for selected channels. The power spectra are plotted 
    for each stage of data processing (raw, filtered, envelope) to facilitate the comparison of 
    signal processing effects on the EEG data.
    
    Parameters
    ----------
    data : dict
        A dictionary containing raw EEG data, channel information, and sing frequency. 
        The relevent keys are 'eeg', 'channels', and 'fs'.
    filtered_data : numpy array
        A 2D numpy array of the filtered EEG data with the same dimensions and channel order as `data['eeg']`.
    envelope : numpy array
        A 2D numpy array of the envelope of the filtered EEG data, obtained through signal processing 
        techniques such as Hilbert transform.
    
    Returns
    -------
    None
    
    Notes
    -----
    The function specifically targets the 'Fz' and 'Oz' channels for analysis but can be easily 
    modified to include additional or different channels. It uses subplots to create a 3-column 
    layout for each channel, where each column corresponds to one of the data stages (raw, filtered, envelope).
    Epoching is performed on the data before computing the power spectrum, which is then converted 
    to decibels for plotting. The function relies on external functions `epoch_ssvep_data` and 
    `get_frequency_spectrum` for epoching and spectrum analysis, respectively.
    
    """

    # label the channels to plot
    channels_to_plot = ['Fz', 'Oz']
    
    # Generate the subplots based on the number of channels
    fig, axs = plt.subplots(len(channels_to_plot), 3, figsize=(15, 10))
    
    # Define epoch parameters 
    epoch_start_time = 0  # in seconds
    epoch_end_time = 2    # in seconds

    # Iterates through the channel's data and sets up the subplot for Raw, Filtered, and Envelope.
    for i, channel in enumerate(channels_to_plot):
        channel_idx = np.where(data['channels'] == channel)[0][0]
        
        # Stage the data for epoching
        for j, stage_data in enumerate([data['eeg'], filtered_data, envelope]):
            # Prepare the data dictionary for epoching
            data_for_epoching = {
                'eeg': stage_data[channel_idx][np.newaxis, :],  # Add an axis to make it 2D array
                'channels': np.array([channel]),
                'fs': data['fs'],
                'event_samples': np.array([0]),  
                'event_durations': np.array([len(stage_data[channel_idx])]),
                'event_types': np.array([1])  # Dummy event type
            }
            
            # Epoch the data with the function epoch_ssvep_data from import_ssvep_data.py
            eeg_epochs, epoch_times, _ = epoch_ssvep_data(data_for_epoching, epoch_start_time, epoch_end_time)
            
            # Obtain the frequency spectrum with the function get_frequency_spectrum from import_ssvep_data.py
            eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, data['fs'])
            
            # Calculate the power spectrum in dB
            power_spectrum = np.mean(np.abs(eeg_epochs_fft) ** 2, axis=0)  # Average over epochs
            power_spectrum_db = 10 * np.log10(power_spectrum) # Conversion to decibels (dB)
            
            # Plot on the graph both the frequency and power spectrum for both channels
            axs[i, j].plot(fft_frequencies, power_spectrum_db[0], label=channel)
            axs[i, j].set_title(f'{channel} - {"Raw" if j == 0 else "Filtered" if j == 1 else "Envelope"}')
            axs[i, j].set_xlabel('Frequency (Hz)')
            axs[i, j].set_ylabel('Power (dB)')
            axs[i, j].set_xlim(0, 60) 
            axs[i, j].legend()

    # Display in a tight layout
    plt.tight_layout()

    # Display the plot
    plt.show()    