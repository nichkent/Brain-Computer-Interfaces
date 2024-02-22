"""
Created on Wed Jan 17, 5:50pm 2024
@authors: Michael Gallo, Nick Kent

eeg_analysis, the purpose of application is to analyize data from a P300 device
to find when a subject has a statistically significant reaction to a target event.
These events are then plotted on spatial maps for easy viewing of specific electrode
reactions. Allows the users of this program to have informative data to further improve on
future uses of the P300 devices.
"""
from matplotlib import pyplot as plt
import numpy as np
import load_p300_data
import plot_p300_erps

#%% Part A


def prepare_epoch_data(subject_number, data_directory='./P300Data/'):
    """
        Prepare EEG data by splitting it into target and non-target epochs.
          Also calculates mean target/non-target epoch data.

        Parameters:
         - subject_number <int> : Subject ID as named on file
         - data_directory <str> : Path to directory where the EEG data files are located.

        Returns:
         - target_erp <np.array>[SAMPLES_PER_EPOCH, NUM_CHANNELS] : ERPs for target events
         - nontarget_erp <np.array>[SAMPLES_PER_EPOCH, NUM_CHANNELS] : ERPs for non-target events
         - erp_times <np.ndarray>[SAMPLES_PER_EPOCH] : time points relative to flashing onset.
         - target_epochs <numpy.ndarray>[NUM_TARGET_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : List of epochs where target letter is included in row/column.
         - nontarget_epochs: <numpy.ndarray>[NUM_NONTARGET_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : List of epochs where target letter is NOT included in row/column.
    """
    # Load eeg experiment results into corresponding arrays
    #   - See function definition for types/dimensions
    eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(subject_number, data_directory)
    # Get indices where rows/columns are flashed and
    #  whether they contain the target letter
    #   - See function definition for types/dimensions
    event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
    # Load data from blocks around each flash event, create array
    #  to represent the time of each element in relation to the flash event.
    #   - See function definition for types/dimensions
    eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)
    # Combine all ERP data to two blocks--one with the average ERP from
    #  all target epochs, and one with the average ERP from all non-target epochs.
    #   - See function definition for types/dimensions
    target_erp, nontarget_erp = plot_p300_erps.get_erps(eeg_epochs, is_target_event)

    # Split the eeg_epochs into target and non-target epochs for further analysis
    target_epochs = eeg_epochs[is_target_event == 1]
    nontarget_epochs = eeg_epochs[is_target_event == 0]

    return target_erp, nontarget_erp, erp_times, target_epochs, nontarget_epochs


#%% Part B


def calculate_se_mean(epochs):
    """
        Calculate Standard Error of the Mean (SEM) for given data.
    
        Parameters:
         - epochs <np.array>[NUM_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS]: 3d array representing an epoch for each event in our data.
             ~ epochs[i][j][k], where i represents the i(th) epoch, 
                                     j represents the j(th) sample in the epoch,
                                     k represents the k(th) channel of data in the epoch.
        Returns:
        - se_mean <np.array>[] : SEM values for each time point across trials.
    """    
    # Use numpy to calculate standard deviation for each time point
    std = np.std(epochs, axis=0)
    
    # Number of trials
    n = epochs.shape[0]
    
    # Calculate the standard error of the mean for confidence intervals
    se_mean = std / np.sqrt(n)
    
    # Return standard error of the epochs
    return se_mean


def plot_confidence_intervals(target_erp, nontarget_erp, erp_times, target_epochs, nontarget_epochs):
    """
        Plots the ERPs on each channel for target and nontarget events.
        Plots confidence intervals as error bars around these ERPs.
        Shows error range for the ERPs.
         
        Params:
         - target_erp <np.array>[SAMPLES_PER_EPOCH, NUM_CHANNELS] : ERPs for target events
         - nontarget_erp <np.array>[SAMPLES_PER_EPOCH, NUM_CHANNELS] : ERPs for non-target events
         - erp_times <np.ndarray>[SAMPLES_PER_EPOCH] : time points relative to flashing onset.
         - target_epochs <numpy.ndarray>[NUM_TARGET_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : List of epochs where target letter is included in row/column.
         - nontarget_epochs: <numpy.ndarray>[NUM_NONTARGET_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : List of epochs where target letter is NOT included in row/column.
             
        Returns:
         - None
    """    
    # Calculate se_mean for both target and nontarget ERPs
    target_se_mean = calculate_se_mean(target_epochs)
    nontarget_se_mean = calculate_se_mean(nontarget_epochs)
    
    # Determine the number of channels from the shape of the data
    num_channels = target_erp.shape[1]
    
    # Define the layout of the subplots. Adusts the number of rows based on the number of channels
    cols = 3
    rows = num_channels // cols + (num_channels % cols > 0)
    
    plt.figure(figsize=(10, rows * 3))
    
    # Iterates through all channels
    for channel_index in range(num_channels):
        # Create a subplot of all the channels for the current subject
        plt.subplot(rows, cols, channel_index + 1)
        
        # Pulls target_erp, nontarget_erp, target_se_mean, and nontarget_se_mean for each channel
        target_erp_channel = target_erp[:, channel_index]
        nontarget_erp_channel = nontarget_erp[:, channel_index]
        target_se_mean_channel = target_se_mean[:, channel_index]
        nontarget_se_mean_channel = nontarget_se_mean[:, channel_index]
        
        # Plot target/nontarget ERPs for the current channel
        plt.plot(erp_times, target_erp_channel, label='Target ERP')        
        plt.plot(erp_times, nontarget_erp_channel, label='Non-Target ERP')
        
        # Fill in 95% confidence interval by adding/subtracting 2*SEM to/from mean ERP.
        plt.fill_between(erp_times, target_erp_channel - 2 * target_se_mean_channel, target_erp_channel + 2 * target_se_mean_channel, alpha=0.2, label='Target +/- 95% CI')
        plt.fill_between(erp_times, nontarget_erp_channel - 2 * nontarget_se_mean_channel, nontarget_erp_channel + 2 * nontarget_se_mean_channel, alpha=0.2, label='Non-Target +/- 95% CI')
        
        plt.xlabel('Time (ms)') # X axis label
        plt.ylabel('Amplitude (µV)') # Y axis label
        plt.title(f'Channel {channel_index}') # Title
        
        if (channel_index == num_channels - 1):
            plt.legend()
    
    # Show plot in a tight layout
    plt.tight_layout()
    #plt.show()


#%% Part C


def bootstrap_p_values(target_epochs, nontarget_epochs, num_iterations=3000):
    """
        Calculate p-values for each time point and channel using bootstrapping.
    
        Parameters:
         - target_epochs <numpy.ndarray>[NUM_TARGET_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : List of epochs where target letter is included in row/column.
         - nontarget_epochs: <numpy.ndarray>[NUM_NONTARGET_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : List of epochs where target letter is NOT included in row/column.
         - num_iterations <int> : number of bootstrapping iterations
    
        Returns:
         - p_values: np.array, shape (SAMPLES_PER_EPOCH, NUM_CHANNELS), p-values for each time point and channel
    """
    # Combine target and non-target epochs for resampling under the null hypothesis
    combined_epochs = np.concatenate((target_epochs, nontarget_epochs), axis=0)
    
    # Difference between target and non-target means, for later use
    observed_diff = np.mean(target_epochs, axis=0) - np.mean(nontarget_epochs, axis=0)
    
    # Initialize an array to hold the bootstrap differences
    bootstrap_diffs = np.zeros((num_iterations, observed_diff.shape[0], observed_diff.shape[1]))
    
    # Total number of epochs for resampling
    num_epochs = combined_epochs.shape[0]
    
    for i in range(num_iterations):
        # Sample with replacement from the combined set of epochs
        bootstrap_epoch_indices = np.random.randint(0, num_epochs, size=num_epochs)
        # Create bootstrap samples for target and non-target
        bootstrap_sample = combined_epochs[bootstrap_epoch_indices]
        bootstrap_target_sample = bootstrap_sample[:len(target_epochs)]
        bootstrap_nontarget_sample = bootstrap_sample[len(target_epochs):]

        # Calculate the difference between the means of the bootstrap samples
        bootstrap_diff = np.mean(bootstrap_target_sample, axis=0) - np.mean(bootstrap_nontarget_sample, axis=0)
        bootstrap_diffs[i] = bootstrap_diff
    
    # Calculate proportion of bootstrap differences at least as extreme as the observed difference
    p_values = np.sum(np.abs(bootstrap_diffs) >= np.abs(observed_diff), axis=0) / num_iterations
    
    return p_values


#%% Part D


from mne.stats import fdr_correction

def plot_confidence_intervals_with_significance(target_erp, nontarget_erp, erp_times, target_epochs, nontarget_epochs, p_values, subject_number):
    """
        Plots the ERPs on each channel for target and nontarget events, including confidence intervals and significant differences marked with dots.

        Parameters:
         - target_erp <np.array>[SAMPLES_PER_EPOCH, NUM_CHANNELS] : ERPs for target events
         - nontarget_erp <np.array>[SAMPLES_PER_EPOCH, NUM_CHANNELS] : ERPs for non-target events
         - erp_times <np.ndarray>[SAMPLES_PER_EPOCH] : time points relative to flashing onset.
         - target_epochs <numpy.ndarray>[NUM_TARGET_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : List of epochs where target letter is included in row/column.
         - nontarget_epochs: <numpy.ndarray>[NUM_NONTARGET_EPOCHS, SAMPLES_PER_EPOCH, NUM_CHANNELS] : List of epochs where target letter is NOT included in row/column.
         - p_values: <np.array>[SAMPLES_PER_EPOCH, NUM_CHANNELS] : p-values for each time point and channel
         - subject_number <int> : Subject ID we are analyzing, used for saving resulting graph image.
        
        Returns:
         - None
    """
    # Calculate se_mean for both target and nontarget ERPs
    target_se_mean = calculate_se_mean(target_epochs)
    nontarget_se_mean = calculate_se_mean(nontarget_epochs)
    
    # Determine the number of channels from the shape of the data
    num_channels = target_erp.shape[1]

    # Define the layout of the subplots. Adusts the number of rows based on the number of channels
    cols = 3
    rows = num_channels // cols + (num_channels % cols > 0)
    plt.figure(figsize=(10, rows * 3))

    # Apply FDR correction to p-values with an alpha threshold of 0.05 and store the corrected p-values.
    _, corrected_p_values = fdr_correction(p_values, alpha=0.05)

    for channel_index in range(num_channels):
        # Create a subplot of all the channels for the current subject
        plt.subplot(rows, cols, channel_index + 1)
        
        # Pulls target_erp, nontarget_erp, target_se_mean, and nontarget_se_mean for each channel
        target_erp_channel = target_erp[:, channel_index]
        nontarget_erp_channel = nontarget_erp[:, channel_index]
        target_se_mean_channel = target_se_mean[:, channel_index]
        nontarget_se_mean_channel = nontarget_se_mean[:, channel_index]

        # Plot target/nontarget ERPs for the current channel
        plt.plot(erp_times, target_erp_channel, label='Target ERP')        
        plt.plot(erp_times, nontarget_erp_channel, label='Non-Target ERP')
        
        # Mark significant differences with dots
        significant_times = erp_times[corrected_p_values[:, channel_index] < 0.05]
        # Flag to ensure we only add the p_value label once
        first_time_labeling_flag = True
        for time in significant_times:
            # Also include label if first addition
            if first_time_labeling_flag:
                plt.plot(time, 0, 'ko', label=r'$p_{FDR}$ < 0.05')
                first_time_labeling_flag = False
            # Otherwise, plot normally
            else:
                plt.plot(time, 0, 'ko')
        
        # Fill in 95% confidence interval by adding/subtracting 2*SEM to/from mean ERP.
        plt.fill_between(erp_times, target_erp_channel - 2 * target_se_mean_channel, target_erp_channel + 2 * target_se_mean_channel, alpha=0.2, label='Target +/- 95% CI')
        plt.fill_between(erp_times, nontarget_erp_channel - 2 * nontarget_se_mean_channel, nontarget_erp_channel + 2 * nontarget_se_mean_channel, alpha=0.2, label='Non-Target +/- 95% CI')
        
        # Add reference lines at x=0 and y=0
        plt.axhline(0, color='black', linestyle='dotted')
        plt.axvline(0, color='black', linestyle='dotted')

        # Add axis labels and title
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (µV)')
        plt.title(f'Channel {channel_index}')
        
        # Only show the legend on the last subplot
        if (channel_index == num_channels - 1):
            plt.legend(loc='lower right')

    plt.tight_layout()

    # Save the plot
    plt.savefig(f'output/subject_{subject_number}_ERP_significance.png')
    # Close plot after saving
    plt.close()


#%%
# Part E
def eval_across_subjects():
    """
        Evaluates significant EEG responses across all subjects and channels, compiles the results, 
        and plots a summary graph. Calls plot_significance_across_subjects method after all calculations.
        
        Parameters:
        - None
    
        Returns:
        - None
    """
    # Initialize a list to hold significant channels and time points across subjects
    significant_channels_timepoints = []

    # Loop through each subject
    for subject_number in range(3, 11):  # Subjects 3 to 10
        # Prepare epoch data
        target_erp, nontarget_erp, erp_times, target_epochs, nontarget_epochs = prepare_epoch_data(subject_number)

        # Calculate p-values using bootstrapping
        p_values = bootstrap_p_values(target_epochs, nontarget_epochs)

        # Plot ERPs with significance and save the plot
        plot_confidence_intervals_with_significance(target_erp, nontarget_erp, erp_times, target_epochs, nontarget_epochs, p_values, subject_number)

        # Update significant_channels_timepoints with results from this subject
        _, corrected_p_values = fdr_correction(p_values, alpha=0.05)
        
        # Boolean array of significant points
        significant = corrected_p_values < 0.05  
        
        # Add to significant_channels_timepoints array
        significant_channels_timepoints.append(significant)

    # After processing all subjects, compile results to find common significant points from the values in significant_channels_timepoints array
    all_significant = np.sum(np.array(significant_channels_timepoints), axis=0)

    # Transpose to shape (8, 384)
    all_significant = all_significant.T

    # Plot the compiled significant points across all subjects
    plot_significance_across_subjects(all_significant, erp_times)


def plot_significance_across_subjects(all_significant, erp_times):
    """
        Plots subjects that have a significant EEG response for each channel at each time point. Saves
        this graph to a figure in output called significance_across_subjects.png.

        Parameters:
        - all_significant <np.ndarray>[NUM_CHANNELS, SAMPLES_PER_EPOCH] : A 2D array where each element 
          at [i, j] represents the count of subjects with a significant response in channel i at time point j.
        - erp_times <np.ndarray>[SAMPLES_PER_EPOCH] : Array of time points for ERP data sampling, relative to 
          the onset of a stimulus.
    """
    # Determine the number of channels
    num_channels = all_significant.shape[0]

    # Define the layout of the subplots
    cols = 3 # Number of columns for the plot
    rows = num_channels // cols + (num_channels % cols > 0) # Number of rows is based on the channels and the columns
    plt.figure(figsize=(10, rows * 3), clear=True) # Create size of the figure
    plt.suptitle('Significance Across Subjects per Channel')

    # Plot each channel on the graph as a subplot
    for channel_index in range(num_channels):
        plt.subplot(rows, cols, channel_index + 1)
        plt.plot(erp_times, all_significant[channel_index, :])
        plt.ylim(-0.1, 4.1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Number of Subjects')
        plt.title(f'Channel {channel_index} Significance Count')
        plt.grid(visible=True)

    plt.tight_layout()
    
    # Save the plot to the output folder
    plt.savefig('output/significance_across_subjects.png')
    
    # .close() to help with run time
    plt.close()

#%% 
# Part F
import plot_topo

def plot_group_median_erp_spatial_distribution():
    """
    Calculates and plots the median ERP distrubtion for both N2 and P3b voltages found in
    the epoch data. Plots each median on a separate spatial map to help visualize the voltages.
    
    Parameters:
        None
    Returns:
        None

    """
    # Define list vars
    group_median_erps = []
    channel_names = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'Oz', 'PO8']
    erp_times_global = None

    # Iterate through subjects 3-10
    for subject_number in range(3, 11):
        # Load epochs and event info using the prepare_epoch_data function
        target_erp, nontarget_erp, erp_times, target_epochs, nontarget_epochs = prepare_epoch_data(subject_number)
        
        # Use erp_times from the first subject as global ERP times
        if erp_times_global is None:# TODO
            erp_times_global = erp_times

        group_median_erps.append(target_erp)
        
        # Plot each subject's ERP data
        # The range for each N2 Median (200, 300) was based off when these voltages are most likely to occur, 200 milliseconds - 300 milliseconds
        # The range for each P3b Median (300, 600) was based off when these voltages are most likely to occur, 300 milliseconds - 600 milliseconds
        for time_range, plot_title in [((200, 300), f"Subject {subject_number} N2 Voltages"), ((300, 600), f"Subject {subject_number} P3b Voltages")]:
            # Convert time_range from milliseconds to seconds
            time_range_seconds = (time_range[0] * 0.001, time_range[1] * 0.001)
            
            # Find indices within the specified time ranges
            # Uses the first subject's erp time (erp_times_global) as a comparator for the range of the graph
            time_indices = np.where((erp_times >= time_range_seconds[0]) & (erp_times <= time_range_seconds[1]))[0]
            
            # Calculate median ERP range using the time indicies found previously
            subject_erp_range = target_erp[time_indices, :].mean(axis=0)
            
            # Use plot_topo function to plot the spatial distribution of ERPs
            plot_topo.plot_topo(channel_names=channel_names, channel_data=subject_erp_range, title=plot_title, save_fig=True, fig_filename=f"output/Subject_{subject_number}_{plot_title}.png")

    # Convert list to a NumPy array for further processing
    group_median_erps_array = np.stack(group_median_erps)

    # Compute the final median ERP across subjects
    final_median_erp = np.median(group_median_erps_array, axis=0)

    # Plot for N2 and P3b ranges
    # The range for each N2 Median (.2, .3) was based off when these voltages are most likely to occur, .2 milliseconds - .3 milliseconds
    # The range for each P3b Median (.3, .6) was based off when these voltages are most likely to occur, .3 milliseconds - .6 milliseconds
    for time_range, plot_title in [((.2, .3), "N2 Median Voltages"), ((.3, .6), "P3b Median Voltages")]:
        # Find indices within the specified time ranges
        # Uses the first subject's erp time (erp_times_global) as a comparator for the range of the graph
        time_indices = np.where((erp_times_global >= time_range[0]) & (erp_times_global <= time_range[1]))[0]

        # Calculate median ERP range using the time indicies found previously
        median_erp_range = final_median_erp[time_indices, :].mean(axis=0)
        
        # Use plot_topo function to plot the spatial distribution of ERPs
        plot_topo.plot_topo(channel_names=channel_names, channel_data=median_erp_range, title=plot_title, save_fig=True, fig_filename=f"output/Group_{plot_title}.png")
