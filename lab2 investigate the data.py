# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:59:00 2024

@author: Jim03121957
"""
import numpy as np
from matplotlib import pyplot as plt
# Import previous module
from load_p300_data import load_training_eeg

# Define subject variable
subject_3 = 3

# Define data directory variable
data_directory = "./P300Data"

# Call the previous module
eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject_3, data_directory)


plt.plot(eeg_time,is_target)
plt.figure()
plt.hist(rowcol_id,bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13],range = [0,13])
'''
#subject 3
(array([58266.,   300.,   300.,   300.,   300.,   300.,   300.,   300.,
          300.,   300.,   300.,   300.,   300.]),
 array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
        13.]),
 <BarContainer object of 13 artists>)

Subject 5 has same number of flashes as subject 3

'''

np.sum(is_target)   #600 true, 61866 false
np.sum(rowcol_id)   #23400, average of row 1to 12 = 6.5 
np.sum(rowcol_id)/6.5  # 3600
#ratio of target to flash should be 1/6, 600/3600

np.size(np.where(rowcol_id[:] == 12))   #indicates that the numer is 12, the 600 must be a error in my use of hist
np.max(eeg_data[:,:])      # subject 3 = 369 ,  subject 5 = 312936
np.min(eeg_data[:,:])      # subject 3 = -132.2 subject 5 = -139976.78549239936
np.mean(eeg_data[:,:])     # subject 3 = 0,     Subjecct 5 = -3.41

np.var(eeg_data[:,:])      # Subject 3 = 83.48 Subject 5 = 122259158.99556248

# Part 2
# Import get_events method
from plot_p300_erps import get_events

# get_events call
event_sample, is_target_event = get_events(rowcol_id, is_target)

# Number aof positive =sum rowcol_id value / divided by duration of flasah 4  / mean 6.5 = 900
Positive_num = np.sum(np.where((np.diff(rowcol_id>0))))  #why 1800 should be 900?
Negative_num = np.sum(np.diff(rowcol_id[:]<0))  #why 1800 should be 900?
No_change = np.sum(np.diff(rowcol_id[:]==0))  # correct for incorrect results above

A =   np.diff(rowcol_id[:]>0)   #wrong
A = np.diff(rowcol_id)>0        #correct
Positive_num = np.sum(np.diff(rowcol_id)>0) #correct 900 
Negative_num = np.sum(np.diff(rowcol_id)<0) #correct 900 
No_change = np.sum(np.diff(rowcol_id)==0) #correct should be 61866-900-900

np.size(event_sample)  # and why is the size of event sample = 900

test_ID = rowcol_id[12500:12665]
Positive_num = np.sum(np.diff(test_ID)>0)  # should be 3
Negative_num = np.sum(np.diff(test_ID)<0)  # should be 3
No_change = np.sum(np.diff(test_ID)==0)  # should be 12665-12500-6 159

np.size(event_sample)
np.sum(is_target)/4  # Divided by four because the flash lasts for 4 sample periods
np.sum(is_target_event)  # this should be 150
#%% Part 3

from plot_p300_erps import epoch_data


(erp_times,eeg_epochs) = epoch_data (eeg_time, eeg_data, event_sample,epoch_start_time = -0.5, epoch_end_time = 1.0)

#%% Part 4

# Import get_erps method
from plot_p300_erps import get_erps

# get_erps call
target_erp, nontarget_erp = get_erps(eeg_epochs, is_target_event)



 # Boolean indexing to extract target and nontarget epochs
 # Include all elements from samples, channels for each is_target_event
target_epochs = eeg_epochs[:, :, is_target_event]  #  axis 2 should be 150
nontarget_epochs = eeg_epochs[:, :, ~is_target_event] #  axis 2 should be 750

 # Calculate mean across epochs for target and nontarget events
 # Axis = 2 to compute the mean across each [sample, channel]
target_erp = np.mean(target_epochs, axis=2)
nontarget_erp = np.mean(nontarget_epochs[:,:,1:150], axis=2) # make sample size the same as target erp= 150
#nontarget_erp = np.mean(nontarget_epochs, axis=2)





#%% Part 6

#%% Investigate subject 5 sensor 5 bad?
data_directory = "./P300Data"
from plot_p300_erps import get_events
from plot_p300_erps import epoch_data
from plot_p300_erps import get_erps
from plot_p300_erps import plot_erps
import matplotlib.pyplot as plt





subject_index = 3
eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject_index, data_directory)
event_sample, is_target_event = get_events(rowcol_id, is_target)
erp_times,eeg_epochs = epoch_data (eeg_time, eeg_data, event_sample,epoch_start_time = -0.5, epoch_end_time = 1.0)
target_erp, nontarget_erp = get_erps(eeg_epochs, is_target_event)
plot_erps(target_erp,nontarget_erp,erp_times,subject_index)
for sensor_index in range (0,8):
    plt.figure()
    plt.plot(eeg_data[sensor_index,:])
    print(max(eeg_data[sensor_index,:]))
    
epoch = 100 # arbitray value
for sensor_index in range (0,8):
    plt.figure()
    plt.plot(eeg_epochs[1,sensor_index,:])
    
    #print(max(eeg_data[sensor_index,:]))   
    
    
    
    
    
    

#%% Investigate subject 4 sensor 5-8 bad?

subject_index = 4
eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject_index, data_directory)
event_sample, is_target_event = get_events(rowcol_id, is_target)
erp_times,eeg_epochs = epoch_data (eeg_time, eeg_data, event_sample,epoch_start_time = -0.5, epoch_end_time = 1.0)
target_erp, nontarget_erp = get_erps(eeg_epochs, is_target_event)
plot_erps(target_erp,nontarget_erp,erp_times,subject_index)
for sensor_index in range (0,8):
    plt.figure()
    plt.plot(eeg_data[sensor_index,:])
    print(max(eeg_data[sensor_index,:]))




