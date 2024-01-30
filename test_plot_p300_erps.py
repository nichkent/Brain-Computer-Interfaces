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

#%% Part 3
from plot_p300_erps import epoch_data

(erp_times,eeg_epochs) = epoch_data (eeg_time, eeg_data, event_sample,epoch_start_time = -0.5, epoch_end_time = 1.0)

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