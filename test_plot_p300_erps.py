# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:05:42 2024

File: test_plot_p300_erps.py
Authors: Nicholas Kent, James Averill
Date: 1/25/2024
Description: This script, test_plot_p300_erps.py, calls all functions in plot_p300_erps.py with subjects 3-10 being displayed for
the plots.
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

#%% Part 3
from plot_p300_erps import epoch_data

(erp_times,eeg_epochs) = epoch_data (eeg_time, eeg_data, event_sample,epoch_start_time = -0.5, epoch_end_time = 1.0)

#%%
# Part 4
# Import get_erps method
from plot_p300_erps import get_erps

# get_erps call
target_erp, nontarget_erp = get_erps(eeg_epochs, is_target_event)



#%% Part 5 Function call

#C:\Users\Jim03121957\OneDrive\Documents\GitHub\BCI


from plot_p300_erps import plot_erps

subject_index = 3

plot_erps(target_erp,nontarget_erp,erp_times,subject_index)

'''
#%% Test by visualizing the data. The following should be removed 

import matplotlib.pyplot as plt 

plt.figure(1,clear=True)
#plt.plot(marker)
#Include a overall title for the plot "Subject"
plt.figtext(.02, .975, 'Test')   #location (x,y),text font characteristics
# Subplot 2 rows 1 column this is the first row

#Subplot 1 
ax1 = plt.subplot(211)                      
plt.plot(erp_times, eeg_epochs[:,0,0])
#plt.plot(eeg_time[plot_range], is_target)
ax1.grid(True, linestyle='-.')
ax1.set_xlabel('Time')
ax1.set_ylabel('eeg raw data (uV)')
plt.ylim(-25, 25)
# share x
plt.tick_params('x', labelbottom=False) # Remove the time values from x axis in this subplot

ax1.set_title('Epoch 0 Channel 0')
         
#Subplot 2 
ax2 = plt.subplot(212, sharex=ax1)
plt.plot(erp_times, eeg_epochs[:,6,899])
ax2.grid(True, linestyle='-.')
plt.ylim(-25, 25)
ax2.set_title('Epoch 900 Channel 7')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('eeg raw data (uV)')
plt.tick_params('x', labelbottom=True)
 
plt.show()
plt.tight_layout()
'''

#%% Part 6 
data_directory = "./P300Data"
from plot_p300_erps import get_events
from plot_p300_erps import epoch_data
from plot_p300_erps import get_erps
from plot_p300_erps import plot_erps
import matplotlib.pyplot as plt

for subject_index in range (3,11):
    eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject_index, data_directory)
    event_sample, is_target_event = get_events(rowcol_id, is_target)
    erp_times,eeg_epochs = epoch_data (eeg_time, eeg_data, event_sample,epoch_start_time = -0.5, epoch_end_time = 1.0)
    target_erp, nontarget_erp = get_erps(eeg_epochs, is_target_event)
    plot_erps(target_erp,nontarget_erp,erp_times,subject_index)
"""
1. The occurance of the repeated up-and-down pattern on many of the channels is likely due to normal background brain activity of the participants.
There were other large magnitude waveforms that could be associated with the placement of the sensors and artfacts that are not relevent to the experiment.
The sources of the artifacts considered were 60hz AC or improper attachment of the sensors that were subject to the participants (i.e. participant moves), neither of these seem feasible due to the time period/frequency of the waveforms.

2. The waveforms that appear to be more pronounced than others tend to be in the occipital lobe. The most noticable change in the occipial lobe appear to happen around
the time of the flash. This indicates that the occipital lobe has a period when it processes the flash and spikes in brain activity, this is then reflected in the spikes in the
parietal lobe sensors as well.

3. The voltage on some channels is likely due to the recognition from the occipital lobe mentioned in the previous response. Once the parietal lobe recieves the information from
the occipital lobe about the flash it will then spike the parietal lobes brain activity.

4. The sensors associated with seeing and processing the visual information coming into the brain (ie. sensors 3-7, 8, and 1). These sensors would pick up the information from the 
flash, process the data, and then the frontal sensor would pick up the participant deciding if the information processed was correct or not. All participants showed similar
results had similar answers when the data was collected correctly however, a few of the subjects (ie 4, 5) appear to have artifacting or extreme variation in their sensor data (More on this available in the world document Lab 2 Part 6.dox).
"""
