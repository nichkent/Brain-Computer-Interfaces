# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:57:49 2024

File: test_load_p300_data.py
Author: Nicholas Kent
Date: 1/16/2024
Description: This script, test_load_p300_data.py, reads through data from a  P300 spelling task for a BCI from Christoph Guger, Shahab Daban, Eric Sellers,
Clemns Holzner, Gunther Krausz, Roberta Carabalona, Furio Gramatica, and Guenter Edliner. The script first unpacks all the data from 
the study and then plots all of the data from a select group of subjects. The data of which specific subject can be modified with the code below.
The section denoted Part 8 pulls the word that each participant spelled during their run of the P300 Speller. It then calculates their characters
per minute and averages them to confirm that it is similar to what the study found.
"""
#%%
# Part 1
# Define variables
subject = 3
data_directory = "./P300Data"
data_file = f"{data_directory}/s{subject}.mat"

#%%
# Part 2
# Import modules
import numpy as np
import loadmat
from matplotlib import pyplot as plt

# Holds all the data
data = loadmat.loadmat(data_file)

# Holds our subject's data
train_data = data["s" + str(subject)]['train']
   
#%% 
# Part 3
# Store the data in numpy arrays
eeg_time = np.array(train_data[0]) # Stores subject time

eeg_data = np.array(train_data[1:9]) # Stores subject data

rowcol_id = np.array(train_data[9], dtype = int) # Stores the id of the row and col of the flashed number

is_target = np.array(train_data[10], dtype = bool) # Confirms if the target was flashed at all

#%%  
# Part 4
    
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


#%%
# Part 5
import load_p300_data as lpd

# Redefine subject for this code block
subject = 3

# Call to lpd's load_training_eeg
eeg_time, eeg_data, rowcol_id, is_target = lpd.load_training_eeg(subject, "./P300Data")
# Call to lpd's plot_raw_eeg
lpd.plot_raw_eeg(eeg_time, eeg_data, rowcol_id, is_target, subject)

#%%
# Part 6
import load_p300_data as lpd

# Define subjects list of all the subjects we want the data for
subjects = [3,4,5,6,7,8,9,10]

# Define the relative path to the data
data_directory = "./P300Data"

# Call to lpd's load_and_plot_all
lpd.load_and_plot_all(data_directory, subjects)

#%%
# Part 7
# Print load_training_eeg docstring
print(f"load_training_egg takes a subject number and a data directory to load the data into variables for manipulation. \nparams: \nsubject : default = 3 -- Subject number \ndata_dictionary : default = './P300Data' -- Relative path to the data \nreturn: \neeg_time \neeg_data \nrowcol_id \nis_target")
# Print plot_raw_eeg docstring
print(f"\nplot_raw_eeg plots the data extracted by load_training_eeg for plotting \nparams: \neeg_time -- Length of the run \neeg_data -- uV at each point 1-8 \nrowcol_id -- Stores the id of the row and col of the flashed number \nis_target -- Confirms if the target was flashed at all \nreturns: none")
# Print load_and_plot_all docstring
print(f"\nload_and_plot_all calls load_training_eeg and plot_raw_eeg for each subject in subjects \nparams: \ndata_directory -- Relative path of the data \nsubjects -- Stores multiple subject numbers to run \nreturns: none")


#%%
# Part 8
import load_p300_data as lpd

# Define subjects
subject1 = 3
subject2 = 4
subject3 = 5
subject4 = 6
subject5 = 7
subject6 = 8
subject7 = 9
subject8 = 10

# Define the relative path to the data
data_directory = "./P300Data"

# Call the function twice to see if it was the same for two different participants
sub1_word, sub1_char_pm = lpd.determine_subject_word(subject1, data_directory)
sub2_word, sub2_char_pm = lpd.determine_subject_word(subject2, data_directory)
sub3_word, sub3_char_pm = lpd.determine_subject_word(subject3, data_directory)
sub4_word, sub4_char_pm = lpd.determine_subject_word(subject4, data_directory)
sub5_word, sub5_char_pm = lpd.determine_subject_word(subject5, data_directory)
sub6_word, sub6_char_pm = lpd.determine_subject_word(subject6, data_directory)
sub7_word, sub7_char_pm = lpd.determine_subject_word(subject7, data_directory)
sub8_word, sub8_char_pm = lpd.determine_subject_word(subject8, data_directory)

# Calculate the average across all participants
char_per_min_total = sub1_char_pm + sub2_char_pm + sub3_char_pm + sub4_char_pm + sub5_char_pm + sub6_char_pm + sub7_char_pm + sub8_char_pm
char_per_min_avg = char_per_min_total / 8

print(f"Subject 1: {sub1_word}")
print(f"Subject 1: {sub1_char_pm} Characters Per Minute")
print(f"Subject 2: {sub2_word}")
print(f"Subject 2: {sub2_char_pm} Characters Per Minute")
print(f"Subject 3: {sub3_word}")
print(f"Subject 3: {sub3_char_pm} Characters Per Minute")
print(f"Subject 4: {sub4_word}")
print(f"Subject 4: {sub4_char_pm} Characters Per Minute")
print(f"Subject 5: {sub5_word}")
print(f"Subject 5: {sub5_char_pm} Characters Per Minute")
print(f"Subject 6: {sub6_word}")
print(f"Subject 6: {sub6_char_pm} Characters Per Minute")
print(f"Subject 7: {sub7_word}")
print(f"Subject 7: {sub7_char_pm} Characters Per Minute")
print(f"Subject 8: {sub8_word}")
print(f"Subject 8: {sub8_char_pm} Characters Per Minute")
print(f"Total Character Per Minute Average: {char_per_min_avg}")

# Multi-line comment
# a) How the data documentation told me
# The documentation's abstract says that the test data the P300 speller was trained on asked the participants to spell the word WATER.
# During the actual runs, the documentation says that the participants where asked to type the word LUCAS.

# b) How the documentation helped me figure out how to confirm it with code
# The documentation told me that I could use rowcol_id and is_target to bypass the usage of the eeg_data to figure out which row and col
# were flashed to the participant and whether it was a target or not. However, I appear to only get LUKAS instead of LUCAS. Since this
# value is so close I believe this to be an error specifically with their P300 Speller data as all participants seemed to spell LUCAS this
# with my way of finding the values which seems to be too great of a coincidence to ignore.