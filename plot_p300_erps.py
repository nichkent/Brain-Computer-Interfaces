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