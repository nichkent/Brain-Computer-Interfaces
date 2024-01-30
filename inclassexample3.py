# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:28:11 2024

@author: nicho
"""
# import packages
import numpy as np
from matplotlib import pyplot as plt

# Make 2s of synthetic data at fs=100 Hz
fs = 100 # Sampling frequency in Hz
t = np.arange(0,2,1/fs) 
sample_count = len(t)
channel_count = 2
data_array = np.zeros([sample_count, channel_count])

# Set channel 0 to switch to 1 at t=1
data_array[t>=1, 0] = 1
# Set channel 1 to be 1 until t = 1.5
data_array[t<=1.5, 1] = 1

plt.figure(1, clear=True)
plt.legend(['Channel 1', 'Channel 2'])


plt.plot(t, data_array)
plt.title('Synthetic EEG data')
plt.xlabel('time (s)')
plt.ylabel('voltage (uV)')

# Print the avg voltage between 1 and 1.5s
average_voltage = np.mean(data_array[(t>1) & (t<1.5), :], axis=0)
print(f'avg voltage: ch1 = {average_voltage[0]}, ch1 = {average_voltage[1]}')

# Multiply by component and plot component activity
w = [-1, 1] # component weight
component_voltage = np.matmul[data_array, w]

# Plot component voltage
plt.plot(t, component_voltage)
plt.legend(['Channel 1', 'Channel 2', 'component'])