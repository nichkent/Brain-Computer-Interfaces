# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:34:59 2024

File: test_import_ssvep_data.py
Authors: Nicholas Kent, 
Date: 2/23/2024
Description: This script, test_import_ssvep_data.py, using the functions in import_ssvep_data.py to perform operations on
data from an SSVEP experiment. This script is split into blocks each relating with one of the functions in import_ssvep_data.py
that will allow the user to visualize the information from the SSVEP experiment before and after the Forier Transform has been 
performed on it.
"""

#%%
# Part 1

# Import the module from import_ssvep_data
from import_ssvep_data import load_ssvep_data

# Subject specification, can be either 1 or 2
subject = 1

# Data directory for the subjects data files
data_directory = './SsvepData'

# Call to load_ssvep_data
# Returns a data_dictionary with the supplied subject's information
data_dict = load_ssvep_data(subject, data_directory)
#%%
# Part 2

# Import the module from import_ssvep_data
from import_ssvep_data import plot_raw_data

# Subject specification, can be either 1 or 2
subject = 1

# Data directory for the subjects data files
data_directory = './SsvepData'

# Specifiy channels that we want plotted from the subject's data
channels_to_plot = ['Fz', 'Oz']

# Call to load_ssvep_data
# Returns a data_dictionary with the supplied subject's information
data_dict = load_ssvep_data(subject, data_directory)

# Call to plot_raw_data
# Returns a given subjet's raw EEG data and event markers for espcified channels
plot_raw_data(data_dict, subject, channels_to_plot)

#%%
# Part 3

# Import the module from import_ssvep_data
from import_ssvep_data import epoch_ssvep_data

# Subject specification, can be either 1 or 2
subject = 1

# Data directory for the subjects data files
data_directory = './SsvepData'

# Define start and end times for the epochs
epoch_start_time = 0
epoch_end_time = 20

# Call to load_ssvep_data
# Returns a data_dictionary with the supplied subject's information
data_dict = load_ssvep_data(subject, data_directory)

# Call to epoch_ssvep_data
# Returns the eeg_epochs, epoch_times, and is_trial_15Hz
eeg_epochs, epoch_times, is_trial_15Hz = epoch_ssvep_data(data_dict, epoch_start_time, epoch_end_time)

# Visualize the outputs
#print("eeg_epochs: ", eeg_epochs)
print("epoch_times: ", epoch_times)
#print("is_trial_15Hz: ", is_trial_15Hz)
#%%
# Part 4

# Import the module from import_ssvep_data
from import_ssvep_data import get_frequency_spectrum

fs = data_dict['fs']

eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)

#%%

# Part 5

# Import the module from import_ssvep_data
from import_ssvep_data import plot_power_spectrum

channels = data_dict['channels']
subject = 2
channels_to_plot = ['Fz', 'Oz']

spectrum_db_12Hz, spectrum_db_15Hz = plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject)


#%%
# Part 6
'''
1. The name for the brain signal that leads to the peaks in 12Hz and 15Hz is called an evoked potential. 
They originate from the visual cortex in the occipital lobe. They originate here due to the visual stimulus 
given to the participant during the experiment but usually just appear when visual stimulus occurs in general. 
The function of the occipital region is to receive and process visual information before it is sent to other lobes 
of the brain where it is further processed for reaction and comprehension. 

2. These integer multiples are called harmonics. We see them in the data because when the brain processes information, 
the data is not received linearly. Thus the brain processes the same information multiple times each with a lesser response 
because the stimulus has been seen before.

3. The most likely cause for this peak is the aforementioned nonlinearity of the brain's response to stimuli. In other studies
 such as On the Quantification of SSVEP Frequency Responses in Human EEG in Realistic BCI Conditions by Kus R. and colleagues, 
 they mentioned that they found “In [4], a representative dependence of SSVEP amplitude on frequency response of one subject 
 exhibits three maxima centered on 15, 31 and 41 Hz. Pastor [5] investigated the EEG oscillatory responses to flicker stimulation
 for selected frequencies in the 5–60 Hz range.” Another possibility is the refresh rate.
 
4. The slight upward bump around 10Hz on some electrodes is likely due to the alpha rhythm. This rhythm is most commonly observed 
in the posterior regions of the brain such as the posterior parietal and occipital lobes. This rhythm typically occurs when someone 
is engaged in focused mental activity such as the SSVEP experiment. The channels it is most prominently observed in are Po1 and Po2 
as well as Pz and Oz. From the lectures we can see that the alpha rhythm overlaps with the Hz of the SSVEP with alpha rhythms occurring 
from 8-12 Hz. While beta waves also overlap the SSVEP rhythms with beta waves occurring from 12-30Hz, these are less likely to be the cause 
because beta waves typically occur during REM sleep and have a much smaller amplitude compared to alpha waves making them much harder to detect 
and thus reducing the likelihood of them appearing as artifacts in the data.
'''
