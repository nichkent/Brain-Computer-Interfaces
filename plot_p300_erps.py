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
    """get_erps calculates the Event-Related Potentials (ERPs) for target and nontarget events.
    Creates and returns two arrays, target_erp and nontarget_erp.
       
       params: 
           eeg_epochs : 3D array of EEG data [samples, channels, epochs].
           is_target_event : Boolean array indicating target events.

       returns: 
           target_erp : 2D array of ERPs for target events [samples, channels].
           nontarget_erp : 2D array of ERPs for non-target events [samples, channels].
    """    
    # Boolean indexing to extract target and nontarget epochs
    # Include all elements from samples, channels for each is_target_event
    target_epochs = eeg_epochs[:, :, is_target_event] 
    nontarget_epochs = eeg_epochs[:, :, ~is_target_event]

    # Calculate mean across epochs for target and nontarget events
    # Axis = 2 to compute the mean across each [sample, channel]
    target_erp = np.mean(target_epochs, axis=2)
    nontarget_erp = np.mean(nontarget_epochs, axis=2)

    # Return target_erp and nontarget_erp
    return target_erp, nontarget_erp



#%% Part 5
def plot_erps(target_erp,nontarget_erp,erp_times):
    '''
    Inputs: 
        
        target_erp - Type[]: 2 D matrix:  [sample_index x sensor channels] eeg data when the flashed row or column contains the target letter.
                                axis 0 mean sampled data   
                                axis 1 eeg sensor channels 
        nontarget_erp - Type[]: 2 D matrix:  [sample_index x sensor channels] associated with flashed rows or columns that do not contain the target letter
                                   axis 0 mean sampled data   
                                   axis 1 eeg sensor channels                      
        erp_times - Type[]: a vector [sample index x 1] of times in seconds relative to the start of a column or row flash.                                  
                                sample index is the same value as the sample index in target_erp[] and nontarget_erp[]
     
    Description: Plots the ERP data in 3 x 3 subplots.  The mean target and non target data is plotted separately on top of each other. 
                 Refernce markers are include to indicte the zero voltge level and to identify the onset of the flashed rows or columns.
                 Plot to figure 1  or ()???
                 Store the figure to a file ???
                 Hard code for 8 figures ???
    Returns: None
 
    '''
    # When I restart kernal need to reidentify directory???

    #C:\Users\Jim03121957\OneDrive\Documents\GitHub\BCI ???

    # Create 3 x 3 subplots consisting of the 8 EEG channels.   
    # Design the subplots so that it works regardless of the number of EEG channels. ???
 
    # Do not assume that pypot has been imported
    import matplotlib.pyplot as plt
    # Determine the number of EEG sensor channels

    # Establish a matrix 3x3 of subplots, all with the same x and y axis, figure 10" wide by 7" high
    # Use constrained layout to prevent overlap  ???

    # how to format the size of the legend???
    #location of the legend ???
    #size of figure x inches by y inches ???
    #
#%% set-up the aesthetics of the plot
    #fig, axs = plt.subplots(3, 3, sharex= 'all', sharey= 'all', figsize = (11, 8), constrained_layout = False, num =1, clear=True)
    fig, axs = plt.subplots(3, 3, sharex= 'all', sharey= 'all', figsize = (11, 8), constrained_layout = False,  clear=True)
    # The subplot functions returns:
            #axs: in this case an array axes objects containing subplot aesthetics
    axs[0,0].set_ylabel('Mean EEG Data (uV)', fontsize= 8)
    # Remove the last subplot (9) only 8 sensors
    axs[2,2].set_axis_off()
    #plt.ylim(-2.5, 2.5)
    #plt.ylim(-100, 100)
    plt.xlim(-.5,1)
    plt.grid(axis='both',color='r', linestyle='-', linewidth=2)
    axs[2,0].set_xlabel('Time (sec)',fontsize = 8)
    axs[2,1].set_xlabel('Time (sec)',fontsize = 8)

    axs[1,2].set_xlabel('Time (sec)',fontsize = 8)
    axs[0,0].set_ylabel('Mean EEG Data (uV)', fontsize= 8)
    axs[1,0].set_ylabel('Mean EEG Data (uV)', fontsize= 8)
    axs[2,0].set_ylabel('Mean EEG Data (uV)', fontsize= 8)

    plt.rc('xtick', labelsize=6) 
    plt.rc('ytick', labelsize=6)
    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.5)

#%% Populate the plot with data
    subject_index = 3           # Hard code starting with subject three
    plot_index = 1              # subplots are identified starting at 1
    for row_index in range(3):
        for col_index in range(3):
            if plot_index <= 8:  # 8 is for now the maximum number of sensors channels to plot
               axs[row_index,col_index] = plt.subplot(3,3,plot_index)   # For now assume 3 x 3                   
      #Plot the data                  
               # using the time values in the erp_times vector, plot the mean eeg signals associated with sensor plot_index
               # erp_times and target_erp provided by the get_erps function, which returns two arrays target and non target arrays
               plt.plot(erp_times, nontarget_erp[:,plot_index-1],label = 'Non-Target')
               #plt.plot(erp_times, target_erp[:,plot_index-1],'--','r',label = 'Target Mean')   # assume sensors range from 0 
               # over plot "hold on" the non target mean sensor values 
               plt.plot(erp_times, target_erp[:,plot_index-1],label = 'Target')
               
               #axs[row_index,col_index].legend('Target',fontsize = 4, loc= 'upper left' )
               #axs[row_index,col_index].legend('Target')
               #plt.plot([0,0],[-10,10],'c','--', label = "Event Ref.") # Event reference
               #plt.plot([-.5,1],[-10,-10],'--', label = "Non-Target Mean" )     # test data 
               plt.tick_params('x', labelbottom=True , labelsize = 8)
               plt.tick_params('y', labelleft=True, labelsize=8)
               
               #Plot the refernce markers
               plt.grid(axis='both',color='b', linestyle='--', linewidth = 0.1)         
               plt.plot([-.5,1],[0,0],'c',label = 'Zero Ref') # zero reference 
               
               # When I include this it mewsses up the y axis tics
               plt.plot([0,0],[-2,1.5],'r', label = 'Flash Ref')  # reference associated with the start of row/col flash
       
              #plt.title(f'EGG Sensor {plot_index}',fontsize=10)
              # place a text box in upper center of subplot
               axs[row_index,col_index]=plt.text(0.4, 0.95, (f'EGG Sensor {plot_index}'), transform=axs[row_index,col_index].transAxes, fontsize=10,
                     verticalalignment='top', bbox=props)      
           
               # Sequence through the sensor channels  
               plot_index += 1 
            else:  # Sensor channel 9 does not exist so don't plot data, instead display the plot legend.
                plt.legend(bbox_to_anchor = (1.4,.9), ncol=1)
                plt.savefig (f'P300_Subject_{subject_index}')
                plt.title (f'Subject {subject_index} Mean Response',fontsize= 10 )
  
#%% Save the figure to a PNG file

  #  fig.savefig ('draft_figure_part_5')   # save the last figure, why does it go into the next level up folder???
        
  #  plt.show()              # not sure if I need this ??? once per session
    
    #plt.tight_layout()  # not sure if I need this 

# END












        