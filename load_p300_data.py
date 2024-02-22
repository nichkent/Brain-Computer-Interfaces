#!/usr/bin/env python3
"""
Created on Wed Jan 17, 5:50pm 2024
@author: magallo

This module contains five functions:
    load_training_eeg(subject, data_directory): Loads training data from filename 
        calculated with the subject number and given data_directory.
    plot_raw_eeg(subject, eeg_time, eeg_data, rowcol_id, is_target): Plots parameters 
        on a 3x1 matplotlib display with subject being displayed in the title, eeg_time 
        as the x-axis in each plot, and the remaining parameters (excluding subject) 
        plotted on the y-axis for each graph.
    load_and_plot_all(data_directory, subjects): Loads training data from filename 
        calculated with the subject number and given data_directory.
    analyze_subject(subject, data_directory): Analyzes given subject using the data 
        directory path. Decodes the word spelled,
        and returns the avg characters typed per minute.
    analyze_all_rc_subjects(data_directory): Analyzes rc subjects 3-10 using the 
        data directory path. Prints each decoded message along with its char-per-min
        speed. Also prints the average # of chars typed per minute over all subjects,
        along with the avg time to type a single character.
    
"""
#%%

import numpy as np
from matplotlib import pyplot as plt
import loadmat as loadmat


#%%
def load_training_eeg(subject, data_directory):
    """
        Loads training data from filename calculated with the subject number and 
        given data_directory.
        
        Args:
            subject <int> : integer representing the subject number.
            data_directory <string> : string representing the file path from script
                directory execution to data files.
        
        Returns:
         - subject <int> : integer representing the subject number.
         - eeg_time <np.array (float)>[TOTAL_SAMPLES] : array containing the time marker for each datapoint.
         - eeg_data <np.array (float)>[TOTAL_SAMPLES, NUM_CHANNELS] : array containing a list of multiple eeg inputs for each datapoint.
         - rowcol_id <np.array (int)>[TOTAL_SAMPLES] : array of integers representing the flashed row/column identifier being flashed for each datapoint.
         - is_target <np.array (bool)>[TOTAL_SAMPLES] : array of booleans representing whether or not the flashed row/column contains the target letter for each datapoint
    """
    # Load data, extract training data
    data_file = f"{data_directory}/s{subject}.mat"
    data = loadmat.loadmat(data_file)
    train_data = data[f"s{subject}"]['train']
    # Time in seconds
    eeg_time = np.array(train_data[0])
    # Data in unknown units (assumed uV)
    eeg_data = np.array(train_data[1:9])
    # Int representing col or row, 1-12
    rowcol_id = np.array(train_data[9], dtype=int)
    # Int 0 or 1, converted to boolean
    is_target = np.array(train_data[10], dtype=bool)
    
    return eeg_time, eeg_data, rowcol_id, is_target


def plot_raw_eeg(subject, eeg_time, eeg_data, rowcol_id, is_target):
    """
        Plots parameters on a 3x1 matplotlib display with subject being 
        displayed in the title, eeg_time as the x-axis in each plot, and the 
        remaining parameters (excluding subject) plotted on the y-axis for each 
        graph.
        
        Args:
         - subject <int> : integer representing the subject number.
         - eeg_time <np.array (float)>[TOTAL_SAMPLES] : array containing the time marker for each datapoint.
         - eeg_data <np.array (float)>[TOTAL_SAMPLES, NUM_CHANNELS] : array containing a list of multiple eeg inputs for each datapoint.
         - rowcol_id <np.array (int)>[TOTAL_SAMPLES] : array of integers representing the flashed row/column identifier being flashed for each datapoint.
         - is_target <np.array (bool)>[TOTAL_SAMPLES] : array of booleans representing whether or not the flashed row/column contains the target letter for each datapoint
    """ 
    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle(f"P300 Speller subject {subject} Raw Data")
    
    # Scope the ranges displayed on the x-axis (eeg_time)
    time_x_min = 48
    time_x_max = 53

    # Row/Col ID subplot
    axs[0].plot(eeg_time, rowcol_id)
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('row/col ID')
    # Scope x-axis
    axs[0].set_xlim(time_x_min, time_x_max)
    axs[0].grid(True)

    # IsTarget subplot
    axs[1].plot(eeg_time, is_target)
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('Target ID')
    # Scope x-axis
    axs[1].set_xlim(time_x_min, time_x_max)
    axs[1].grid(True)

    # EEG Data subplot
    axs[2].plot(eeg_time, eeg_data.T)
    axs[2].set_xlabel('time (s)')
    axs[2].set_ylabel('Voltage (uV)')
    # Scope our plot
    axs[2].set_xlim(time_x_min, time_x_max)
    axs[2].set_ylim(-25, 25)
    axs[2].grid(True)
    
    plt.tight_layout()

    # Save plot to file
    plt.savefig(f"P300_S{subject}_training_rawdata.png")

    # Show plot
    plt.show()
  

#%%

def load_and_plot_all(data_directory, subjects):
    """
        Loads training data from filename calculated with the subject number and given data_directory.
        
        Args:
         - data_directory <string>: string representing the file path from script directory execution to data files.
         - subjects <int[]> : List ofinteger representing the subject number.
         
        Returns:
         - None
    """
    # Iterate through each subject
    for subject_number in subjects:
        # Load training data into variables from passing our subject and data directory into load_training_eeg()
        eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject_number, data_directory)
        # Use our plotting function and our new variables to plot the variables
        plot_raw_eeg(subject_number, eeg_time, eeg_data, rowcol_id, is_target)
    

#%%

def analyze_subject(subject, data_directory):
    """
        Analyzes given subject using the data directory path. Decodes the word spelled,
          and returns the avg characters typed per minute.
        
        Args:
         - subject <int> : integer representing the subject number.
         - data_directory <string> : string representing the file path from script directory execution to data files.
        
        Returns:
         - chars_per_min <double> : Average number of characters typed per minute in given data.
    """
    # Generate 6x6 2d list to reference with our data
    character_matrix = [['A', 'B', 'C', 'D', 'E', 'F'],
                        ['G', 'H', 'I', 'J', 'K', 'L'],
                        ['M', 'N', 'O', 'P', 'Q', 'R'],
                        ['S', 'T', 'U', 'V', 'W', 'X'],
                        ['Y', 'Z', '0', '1', '2', '3'],
                        ['4', '5', '6', '7', '8', '9']]  
    # Load in relevant training variables
    eeg_time, _, rowcol_id, is_target = load_training_eeg(subject, data_directory)
    # Temp variable to store message as it develops
    message = ""
    # Initialize numpy arrays for counts
    row_id_counts = np.zeros(6)
    col_id_counts = np.zeros(6)
    # Store number of times a correct row
    flash_count = 0
    
    for index in range(len(rowcol_id)):
        # Ensure no duplicate letters are recorded for the same flash
        if is_target[index] == 1 and is_target[index - 1] != 1:
            # Increment number of correct flashes
            flash_count += 1
    
            # Count the rowcol_id
            current_rowcol_id = rowcol_id[index]
            if current_rowcol_id > 6:
                row_id_counts[current_rowcol_id - 7] += 1
            elif current_rowcol_id > 0:
                col_id_counts[current_rowcol_id - 1] += 1
    
            # If each row and column has been flashed 15 times
            if flash_count == 30:
                # Get the flashed row/col indices
                row_id = np.argmax(row_id_counts)
                col_id = np.argmax(col_id_counts)
                message += character_matrix[row_id][col_id]

                # Reset row counts, flash count, start_time...
                row_id_counts = np.zeros(6)
                col_id_counts = np.zeros(6)
                flash_count = 0
    
    # Print
    print(f"Message: {message}")
    # Calculate minutes passed in experiment
    mins_in_experiment = (eeg_time[-1] - eeg_time[0]) / 60
    chars_per_min = len(message) / mins_in_experiment
    print(f"Characters-per-minute: {chars_per_min}\n")
    return chars_per_min

def analyze_all_rc_subjects(data_directory):
    """
        Analyzes rc subjects 3-10 using the data directory path.
        Prints each decoded message along with its char-per-min speed.
        Also prints the average # of chars typed per minute over all subjects,
        along with the avg time to type a single character.
        
        Args:
         - data_directory <string> : string representing the file path from script directory execution to data files.
    
        Returns:
         - None
    """
    # Store each avg type speed in the list below
    avg_type_speed = []
    for subject_num in range(3,11):
        avg_type_speed.append(analyze_subject(subject_num, data_directory))
    
    # Calculate avg number of chars typed per min from all subjects
    avg_chars_per_min = np.average(avg_type_speed)
    # Print Results
    print("Characters typed per minute: " + str(avg_chars_per_min))
    print("Avg time to type one character: " + str(60/avg_chars_per_min))