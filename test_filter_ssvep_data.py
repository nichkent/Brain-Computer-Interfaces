# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:28:34 2024

File: test_filter_ssvep_data.py
Authors: Nicholas Kent, Alaina Birney
Date: 3/7/2024
Description: This script, test_filter_ssvep_data.py,
"""

#%%
# Part 1

# Import load_ssvep_data from import_ssvep_data.py
from import_ssvep_data import load_ssvep_data

# Define vars for load_ssvep_data
subject = 1
data_directory='./SsvepData'

# Load the data into a data dictionary called data for later use
data = load_ssvep_data(subject, data_directory)

#%%
# Part 2

# Import make_bandpass_filter from filter_ssvep_data.py
from filter_ssvep_data import make_bandpass_filter

# 12 Hz var definitions
low_cutoff = 11
high_cutoff = 13
filter_type = "hann"
filter_order = 1000
fs = data['fs']

# For 12 Hz, returns the filter coefficients for 12hz
filter_coefficients_b_12hz = make_bandpass_filter(low_cutoff, high_cutoff, filter_type, filter_order, fs)

# 15 Hz var definitions
low_cutoff = 14
high_cutoff = 16
filter_type = "hann"
filter_order = 1000
fs = data['fs']

# For 15 Hz, returns the filter coefficients for 15hz
filter_coefficients_b_15hz = make_bandpass_filter(low_cutoff, high_cutoff, filter_type, filter_order, fs)


print(filter_coefficients_b_12hz)
print(filter_coefficients_b_15hz)
'''
A) For the 12 Hz on the fc=[14, 16] graph for 15Hz filter at 12hz signals was -44.6 dB. And for the 15 Hz on the 
fc=[11, 13] graph for 12 Hz filter at 15 Hz signals was -39.6 dB. These findings show that the hann filter is very
effective at blocking frequencies outside of its designated passband with both the 15 Hz filter almost entirely
blocking out 12 Hz signals with the same results for the 12 Hz filter for 15 Hz signals.

B) The order of a filter has a direct effect on its characteristics:

A HIGHER order filter generally has a steeper roll-off, meaning that it can better distinguish between 
frequencies inside and outside the passband. This results in better separation of desired and undesired frequencies.
A LOWER order filter will have a more gradual roll-off and may allow more of the adjacent frequencies into the passband.

With impulse response, a HIGHER order filter will generally result in a longer impulse response, meaning that it takes
more time for the filter to "settle". This can cause more delay in the signal processing.

Increasing the filter order will narrow the transition band between the passband and stopband but will make the filter 
more computationally expensive and introduce more delay. Decreasing the filter order will do the opposite. The transition 
band becomes wider, requiring less computation and introducing less delay, but at the cost of less sharp separation 
between the filtered and unfiltered frequencies.
'''

#%% Part 3: Filter the EEG Signals
from filter_ssvep_data import filter_data

# filter data with band-pass filter to capture 12hz oscillations
filtered_12hz = filter_data(data, filter_coefficients_b_12hz)

# filter data with band-pass filter to capture 15hz oscillations
filtered_15hz = filter_data(data, filter_coefficients_b_15hz)

#%% Part 4: Calculate the envelope
from filter_ssvep_data import get_envelope

# get 15hz envelope for filtered data for electrode Oz
envelope_15hz = get_envelope(data, filtered_15hz, "Oz", ssvep_frequency="15 Hz")

# get 12hz envelope for filtered data for electrode Oz
envelope_12hz = get_envelope(data, filtered_12hz, "Oz", ssvep_frequency="12 Hz")

#%% Part 5: Plot the Amplitudes
from filter_ssvep_data import plot_ssvep_amplitudes

dif = plot_ssvep_amplitudes(data, envelope_12hz, envelope_15hz, "Oz","12hz", "15hz", 1)

"""
What do the two envelopes do when the stimulation frequency changes?
For subject 1, on average, the 12 Hz envelope displays higher amplitude than
the 15 Hz envelope during the 12 Hz event. When the frequency changes to 15 Hz,
there is no longer a clear difference seen in the amplitudes of the 12 Hz 
and 15 Hz envelopes.

How large and consistent are those changes?


Are the brain signals responding to the events in the way you'd expect?
"""