# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:17:04 2024

@author: Michael Gallo, Nick Kent

test_eeg_analysis, the purpose of this application is to be a frontend for running eeg_analysis.
All functions is eeg_analysis should be called from this file for accurate analysis.
"""
#%%
# Part A
from eeg_analysis import prepare_epoch_data

# Define subject number
subject_number = 3

# Prepare epoch data
target_erp, nontarget_erp, erp_times, target_epochs, nontarget_epochs = prepare_epoch_data(subject_number)


#%%
# Part B
from eeg_analysis import plot_confidence_intervals

# Plot ERPs with confidence intervals
plot_confidence_intervals(target_erp, nontarget_erp, erp_times, target_epochs, nontarget_epochs)

#%%
# Part C

from eeg_analysis import bootstrap_p_values

p_values = bootstrap_p_values(target_epochs, nontarget_epochs)

#%%
# Part D
from eeg_analysis import plot_confidence_intervals_with_significance

subject_number = 3

plot_confidence_intervals_with_significance(target_erp, nontarget_erp, erp_times, target_epochs, nontarget_epochs, p_values, subject_number)

#%%
# Part E
from eeg_analysis import eval_across_subjects

eval_across_subjects()

#%%
# Part F
from eeg_analysis import plot_group_median_erp_spatial_distribution

plot_group_median_erp_spatial_distribution()

