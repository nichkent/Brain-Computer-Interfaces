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
envelope_15hz = get_envelope(data, filtered_15hz, "Fz", ssvep_frequency="15 Hz")

# get 12hz envelope for filtered data for electrode Oz
envelope_12hz = get_envelope(data, filtered_12hz, "Fz", ssvep_frequency="12 Hz")

#%% Part 5: Plot the Amplitudes
from filter_ssvep_data import plot_ssvep_amplitudes

avg_dif_12hz, avg_dif_15hz = plot_ssvep_amplitudes(data, envelope_12hz, envelope_15hz, "Fz","12hz", "15hz", 1)

# the following print statement was written to aid in answering the questions below
print(f"Average difference (12hz envelope - 15 hz envelope)for 12 hz trials: {avg_dif_12hz}")
print(f"Average difference (12hz envelope - 15 hz envelope)for 15 hz trials: {avg_dif_15hz}")

"""
What do the two envelopes do when the stimulation frequency changes?
For subject 1, on average, the 12 Hz envelope displays higher amplitude than
the 15 Hz envelope during the 12 Hz event. When the frequency changes to 15 Hz,
the difference in amplitudes of the 12 Hz and 15 Hz envelopes appears to be
less drastic. However, the 12 Hz envelope still looks to display higher amplitude
than the 15 Hz envelope, on average, during the 15 Hz trials.

How large and consistent are those changes?
The changes are not very large; on average, the 12 Hz envelope is approximately 
0.97697 uV greater in amplitude than the 15 Hz envelope during the 12Hz events
while the 12 Hz envelope is, on average, approximately 0.25413 uV greater in 
amplitude than the 15 Hz envelope during the 15 Hz events. The changes are relatively
consistent in that the 12 Hz envelope is consistently greater in amplitude than 
the 15 Hz envelope during 12 Hz events. However, there is some fluctuation 
during the 15 Hz events; although the 12 Hz envelope displays higher amplitude
on average, occasionally the amplitude of the 15 Hz envelope surpasses the 12
Hz envelope.

Are the brain signals responding to the events in the way you'd expect? 
Although the response does not seem very strong, this is a relatively expected
result if the subject was focused on the 12 Hz target. We should expect 
to see higher amplitude for the envelope representing the frequency
corresponding to the target that the subject was focused on during the corresponding
event, meaning we should see a higher amplitude in the 12 Hz envelope during the 
12 Hz events.

Check some other electrodes- which electrodes respond in the same way and why?
Because SSVEP activity is reflected in the primary visual cortex, it would make
sense for similar responses to occur in elecrodes O1, O2, and Oz. If the 
stimulus is on the left side, activity will be most strongly reflected in O2 while
if the stimulus is on the right side, activity will be most strongly reflected in
O1. 
When the activity is checked for electrode O1, the result is similar to 
results for electrode Oz, but differences are present; for both the 12 Hz and 
15 Hz events, the 12 Hz envelope is higher in amplitude. However, in electrode 
O1, on average, the difference between the 12 Hz and 15 Hz envelopes is less 
drastic than in electrode Oz during 12 Hz events (the average difference of the 
12 Hz envelope minus the 15 Hz envelope is approximately 0.89052 uV during 12 
Hz events for electrode O1) and the difference between the 12 Hz and 15 Hz 
envelopes is more drastic than in electrode Oz during 15 Hz events (the average 
difference of the 12 Hz envelope minus the 15 Hz envelope is approximately 
0.40429 uV during 15 Hz events for electrode O1). 
When the activity is checked for electrode O2, the result is similar to results
for electrode Oz, but more drastic. For both the 12 Hz and 15 Hz events, the 12 
Hz envelope is higher in amplitude. Compared to results for electrode Oz, the 
12 Hz envelope is higher in amplitude during the 12 Hz events (the average 
difference was found to be approximately 0.99781 in this case) and the difference
between the 12 Hz envelope and 15 Hz envelope is less extreme during the 15 Hz 
events (the average difference was found to be approximately 0.22784 in this case). 
These results suggest that the stimulus was on the left side of the subject's 
visual field.
Electrode Fz was also checked. Here, it was again found that the 12 Hz envelope
was higher in amplitude during 12 Hz and 15 Hz events. However, the average difference
was found to be approximately 0.081553 during 12 Hz events while the average
difference was found to be approximately 0.07784 during 15 Hz events. The weak
responses during both events and relatively similar values for 12 Hz and 15 Hz
events align well with our understanding of SSVEPs; as electrode Fz does not
measure activity from the visual cortex, we should not expect to see clear results
for an SSVEP experiment from this electrode.
"""