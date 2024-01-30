# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:04:47 2024

File: plot_p300_erps.py
Authors: Nicholas Kent
Date: 1/25/2024
Description:
"""
# Imports
import numpy as np
from matplotlib import pyplot as plt

#%%
# Part 2
def get_events(rowcol_id, is_target):  
    """get_events this function uses rowcol_id to identify moments when
    a new event occurs. A new event starts when the values in rowcol_id switch from 0 to any number 1-12.
    Creates and returns two arrays, event_sample and is_target_event.
       
       params: 
           int[] rowcol_id : row/col id of the letter being flashed on screen
           bool[] is_target : true if the letter was a letter the participant was supposed to type, false otherwise
       returns: 
           int[] event_sample : An array of indicies where event IDs in rowcol_id move upward. Indicates a start of a new event.
           bool[] is_target_event : An array indicating for each index in event_sample whether the corresponding event was a target in the is_target array.
    """
    # Create an array of the samples when the event ID went up (flashes)
    # Add 1 to the index to account for the forward difference
    event_sample = np.array(np.where(np.diff(rowcol_id) > 0)[0] + 1)
    
    # Use event_sample to index the is_target array
    is_target_event = is_target[event_sample]
            
    # Return event_sample and is_target_event
    return event_sample, is_target_event

#%% Part 3 

def epoch_data (eeg_time, eeg_data, event_sample, epoch_start_time = -0.5, epoch_end_time = 1.0):
     '''
      Inputs:
      
      Purpose:
          
      Outputs: 
      1) np.array containing the Epoch Data which are windowed snap shots of the data around events. 
      The Epoch data is a A three dimensional matrix; axis (0) consists of the raw eeg sensor data, axis (1) consists of the eight eeg sensor channels 
      and axis (2) epoch or realization - each of the identified events (i.e. when a row or column flashes)
      2) Relative time associated with each event 
      
         
      # how did this get here???
    
      Parameters
      ----------
      eeg_time : TYPE: vector of floats.  
          DESCRIPTION. Time associated with sampling off eeg data
      eeg_data : TYPE: Matrix with 2 dimensions
          DESCRIPTION. Axis(0) Associated with sensor, axis(1) raw data
      event_sample : TYPE vector of int
          DESCRIPTION. Index into eeg_data associated when a row or column was flashed
      epoch_start_time : TYPE, optional
          DESCRIPTION. The default is -0.5 (sec).
      epoch_end_time : TYPE, optional
          DESCRIPTION. The default is 1.0 (sec).
    
      Returns
      -------
      eeg_epochs: Type:
                  Description: Matrix with three dimensions.  First axis(0) raw eeg windowed sample data, 
                      second axis(1) each of the epochs, third axis(2) eeg sensors total
      erp_times:  Type: vector of floats 
                  Description: Values represent the relative time to the target event associated with each of the 
                  eeg_epochs data samples.
       
      Assumptions
      Fixed sample time for all the data
      '''    
     # Calaculate the sample period based upon time between two arbritrary samples
     ts = eeg_time[10]-eeg_time[9]
     fs = 1/ts
     
     # The total number of samples = sample window (sec) * fs (samples/sec)
     number_of_samples = int((epoch_end_time - epoch_start_time)*fs +1)
   
    # number_of_samples_plus_one = int(((epoch_end_time - epoch_start_time) * fs) + 1)    # is the +1 needed???? is this the best way???
     erp_times = np.arange(epoch_start_time,(epoch_end_time+1/fs),1/fs)
    
     
    # Total number of epochs or realizations associated with rows or columns flashing
     number_of_epochs = event_sample.size
   
    # Use eeg_data to determine the total number of sensors.
     number_of_channels = len (eeg_data)       
   
     # Declare and intialize to zero the three 3 matrix 
     eeg_epochs = np.zeros([number_of_samples,number_of_channels,number_of_epochs])

     for epoch_index in range (0, number_of_epochs):
         for channel_index in range (0,number_of_channels):  #zero based
             for sample_index in range(0,number_of_samples):
                 #print(f'Epoch = {epoch_index}, Channel = {channel_index} Sample = {sample_index}')
                 # Index into the eeg_data sample values needs to be adjusted for the window start and stop times
                 # sample index for eeg_epochs runs from 0 to the total number of samples
                 # The data contained in egg_data is adjusted based upon the index into event samples
                 # eeg_epoch[0] corrsponds to event_sample[epoch_index] + (epoch_start_time*fs) (note epoch_start_time a negative value)
                # epoch_index = 0
                # epoch_start_time = -0.5
                 start = int(event_sample[epoch_index] + (epoch_start_time*fs))
                 eeg_epochs[sample_index,channel_index,epoch_index]= eeg_data[channel_index,start+sample_index]
                    
     return (erp_times,eeg_epochs)
 
#%%
# Part 4
def get_erps(eeg_epochs, is_target_event):
    """get_erps this function uses eeg_epochs along with is_target_event to find when a target event
     occurs. 
    Creates and returns two arrays, target_erp and nontarget_erp.
       
       params: 
           int[] eeg_epochs : 
           bool[] is_target_event : 
       returns: 
           int[] target_erp : 
           bool[] nontarget_erp : 
    """
    # Use boolean indexing to extract ERPs for the target and nontarget trails
    target_erp = # True boolean indexing
    nontarget_erp = # False boolean indexing
    
    
    return target_erp, nontarget_erp