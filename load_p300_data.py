# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:57:27 2024

File: test_load_p300_data.py
Author: Nicholas Kent
Date: 1/16/2024
Description: This script, load_p300_data.py, reads through data from a  P300 spelling task for a BCI from Christoph Guger, Shahab Daban, Eric Sellers,
Clemns Holzner, Gunther Krausz, Roberta Carabalona, Furio Gramatica, and Guenter Edliner. The script first unpacks all the data from 
the study and then plots all of the data from a select group of subjects. The data of which specific subject can be modified with the code below.
The section denoted Part 8 pulls the word that each participant spelled during their run of the P300 Speller. It then calculates their characters
per minute and averages them to confirm that it is similar to what the study found.
"""
#%%
# Part 5
### Function load_training_eeg
def load_training_eeg(subject = 3, data_directory = "./P300Data"):
    """load_training_egg takes a subject number and a data directory to load the data into variables for manipulation
       params: 
           int subject : default = 3 -- Subject number
           str data_dictionary : default = "./P300Data" -- Relative path to the data
       return: 
           float[] eeg_time -- time recorded at each data step
           float[] eeg_data -- data recorded at each time step
           int[] rowcol_id -- row/col id of the letter being flashed on screen
           bool[] is_target -- true if the letter was a letter the participant was supposed to type, false otherwise
    """
    
    # Part 1
    data_file = f"{data_directory}/s{subject}.mat"
    
    # Part 2
    # Import modules
    import numpy as np
    import loadmat

    # Holds all the data
    data = loadmat.loadmat(data_file)

    # Holds our subject's data
    train_data = data["s" + str(subject)]['train']
    
    # Part 3
    # Store the data in numpy arrays
    eeg_time = np.array(train_data[0]) # Stores subject time

    eeg_data = np.array(train_data[1:9]) # Stores subject data

    rowcol_id = np.array(train_data[9], dtype = int) # Stores the id of the row and col of the flashed number

    is_target = np.array(train_data[10], dtype = bool) # Confirms if the target was flashed at all
    
    # Return call
    return eeg_time, eeg_data, rowcol_id, is_target



def plot_raw_eeg(eeg_time, eeg_data, rowcol_id, is_target, subject):
    """plot_raw_eeg plots the data extracted by load_training_eeg for plotting
       params: 
           float[] eeg_time -- Length of the run
           float[] eeg_data -- uV at each point 1-8
           int[] rowcol_id -- Stores the id of the row and col of the flashed number
           bool[] is_target -- Confirms if the target was flashed at all
       returns: none
    """
    # Part 4
    # Import modules
    from matplotlib import pyplot as plt
    
    # Have a different pop-up for each subject
    plt.figure(subject)
    
    # Plot 1 rowcol_id
    plt.subplot(3, 1, 1)
    plt.plot(eeg_time, rowcol_id)
    plt.title('P300 Speller subject ' + str(subject) +' Raw Data')
    plt.xlim(48, 53) # Limits the time from 0-5 seconds
    plt.xlabel('time (s)')
    plt.ylabel('row/col ID')
    plt.grid()

    # Plot 2 is_target
    plt.subplot(3, 1, 2)
    plt.plot(eeg_time, is_target)
    plt.xlim(48, 53) # Limits the time from 0-5 seconds
    plt.xlabel('time (s)')
    plt.ylabel('Target ID')
    plt.grid()

    # Plot 3 eeg_time
    plt.subplot(3, 1, 3)
    plt.plot(eeg_time, eeg_data.T)
    plt.xlim(48, 53) # Limits the time from 0-5 seconds
    plt.ylim(-25, 25) # Limits the voltage to -25 and 25 uV
    plt.xlabel('time (s)')
    plt.ylabel('voltage (uV)')
    plt.grid()

    # Plot tight_layout and display
    plt.tight_layout()
    plt.show()

    # Save the plot to the current directory
    plt.savefig('P300_S' + str(subject) +'_training_rawdata.png')
    

# Part 6
### Function load_and_plot_all
def load_and_plot_all(data_directory, subjects):
    """load_and_plot_all calls load_training_eeg and plot_raw_eeg for each subject in subjects
       params:
           str data_directory -- Relative path of the data
           int[] subjects -- Stores multiple subjects in an int list
       returns: none
    """
    
    # For loop to run the functions on each subject
    for subject in subjects:
        # Call to load_training_eeg
        eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject, data_directory)
        
        # Call to plot_raw_eeg
        plot_raw_eeg(eeg_time, eeg_data, rowcol_id, is_target, subject)
        
        
#%%
# Part 8
# Write a function to determine what the subjects were asked to type using the truth data
def determine_subject_word(subject, data_directory):
    """determine_subject_word finds the word that each subject typed from rowcol_id and is_target. Determines the characters
        per minute of each participant's run.
       params: 
           int subject : default = 3 -- Subject number
           str data_dictionary : default = "./P300Data" -- Relative path to the data
       return: 
           str spelled_word -- The word the participant spelled in their run
           float char_per_min -- The characters per minute each participant typed at
    """
    # Load data for subject
    eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject)
    
    # Define vars
    spelled_word = ""
    row_id = None
    col_id = None
    last_added_letter_pair = (None, None)  # Track last row-column pair added so there are no duplicate letters in a row
    times_flashed = 0 # Keeps track of the number of times a row and column has been flashed
    MAX_FLASHED_NUM = 30 # Max number that times_flashed can reach

    # Artificial Speller Matrix based on the documentation
    speller_matrix = [['A', 'B', 'C', 'D', 'E', 'F'],
                      ['G', 'H', 'I', 'J', 'K', 'L'],
                      ['M', 'N', 'O', 'P', 'Q', 'R'],
                      ['S', 'T', 'U', 'V', 'W', 'X'],
                      ['Y', 'Z', '0', '1', '2', '3'],
                      ['4', '5', '6', '7', '8', '9']]
    
    # For loop to iterate through rowcol_id
    for i in range(len(rowcol_id)):
        # Only look at values that have been confirmed as a target
        if is_target[i]:
            
            # Determine if it's a row or a column
            if rowcol_id[i] < 7:  # It's a row
                
                # Reset times_flashed if the current row is not equal to the new row
                if row_id != rowcol_id[i] - 1:
                    times_flashed = 0
                    
                # Set the current row. Sub 1 for python indexing
                row_id = rowcol_id[i] - 1
                
            else:  # It's a column
            
                # Reset times_flashed if the current column is not equal to the new column
                if col_id != rowcol_id[i] - 7:
                    times_flashed = 0
                    
                # Set the current column. Sub 6 for column, sub 1 extra for python indexing
                col_id = rowcol_id[i] - 7

            # Increment times_flashed if both row and column are set
            if row_id is not None and col_id is not None:
                times_flashed += 1

                # Check if it's time to add the letter and if the pair is new
                if times_flashed == MAX_FLASHED_NUM and (row_id, col_id) != last_added_letter_pair:
                    spelled_word += str(speller_matrix[col_id][row_id])
                    
                    # Update the last added pair
                    last_added_letter_pair = (row_id, col_id)  
                    
                    # Reset for next letter
                    row_id = None
                    col_id = None
                    times_flashed = 0

    ### Calculate words per minute
    # Find the total minutes of the current run
    total_minutes = eeg_time[-1] / 60
    
    # Determine the characters per minute
    char_per_min = len(spelled_word) / total_minutes
    
                    
    # Return spelled word and words per minute
    return spelled_word, char_per_min