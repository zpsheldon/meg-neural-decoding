import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lime_explanation import apply_perturbation_to_ecg, perturb_mean  

def plot_class_distribution(labels, title="Class Distribution"):
    """
    Plots the distribution of classes using a bar chart, with specific colors for each class.
    
    Parameters:
    - labels (pd.Series): A pandas Series containing class labels.
    - title (str): Title for the plot.
    """
    # Define specific colors for each class
    class_colors = {1: "r", 2: "g", 3: "b", 4: "k"}
    
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=labels)
    ax.set_title(title)
    
    # Get unique classes and their counts
    class_counts = labels.value_counts().sort_index()

    # Iterate over the unique classes and set the colors for each bar
    for i, class_id in enumerate(class_counts.index):
        ax.patches[i].set_color(class_colors[class_id])

    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)  # Rotate class labels to avoid overlap, if necessary
    plt.show()



def plot_sample_signals(ecg_features, ecg_labels):
    """
    Plots one sample signal from each class in the dataset.
    
    Parameters:
        ecg_features (DataFrame): The features of the ECG dataset, where each row is a signal.
        ecg_labels (Series): The labels for the dataset, indicating the class of each signal.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 4))

    unique_classes = ecg_labels.unique()
    for class_ in unique_classes:
        sample_index = ecg_labels[ecg_labels == class_].index[0]
        if class_ == 1:
            plt.plot(ecg_features.loc[sample_index, :], label=f"Class {class_}", color="r")
        elif class_ == 2:
            plt.plot(ecg_features.loc[sample_index, :], label=f"Class {class_}", color="g")
        elif class_ == 3:
            plt.plot(ecg_features.loc[sample_index, :], label=f"Class {class_}", color="b")
        elif class_ == 4:
            plt.plot(ecg_features.loc[sample_index, :], label=f"Class {class_}", color="k")

    plt.title("Sample ECG Signal from Each Class")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend(title="ECG Classes")
    plt.show()

def plot_segmented_ecg(instance_ecg, slice_width):
    """
    Plots an ECG signal and its segments.

    Parameters:
        instance_ecg (np.ndarray): The ECG signal instance to plot.
        slice_width (int): The width of each slice in the segmented ECG signal.
    """
    n_samples = instance_ecg.shape[-1]
    n_channels = instance_ecg.shape[0]
    plt.figure(figsize=(12, 3))
    for chan in range(n_channels):
        plt.plot(np.arange(n_samples), instance_ecg[chan])
    num_slices = n_samples // slice_width
    
    for i in range(1, num_slices):
        plt.axvline(x=i * slice_width, color='r', linestyle='--')

    plt.title('Segmented the instance ECG signal')
    plt.xlabel('Time Index')
    plt.ylabel('Signal Amplitude')
    plt.show()


def plot_perturbed_ecg(original_ecg, perturbed_ecg, perturbation, num_slices, title='ECG Signal with Perturbation'):
    """
    Plots the original and perturbed ECG signals with slices and deactivated segments highlighted.

    Parameters:
    - original_ecg (np.ndarray): The original ECG signal.
    - perturbed_ecg (np.ndarray): The perturbed ECG signal after applying the perturbation.
    - perturbation (np.ndarray): The perturbation vector used to modify the ECG signal.
    - num_slices (int): The total number of segments the ECG signal is divided into.
    - title (str): The title for the plot. Optional.
    """
    total_length = len(original_ecg)
    slice_width = total_length // num_slices

    plt.figure(figsize=(12, 6))

    # Plot original ECG signal with slices and deactivated segments highlighted
    plt.subplot(2, 1, 1)
    plt.plot(original_ecg, label='Original ECG Signal', color='black')
    plt.title(f'Original {title}')
    for i in range(num_slices):
        start_idx = i * slice_width
        end_idx = min((i + 1) * slice_width, len(original_ecg))
        plt.axvline(x=start_idx, color='r', linestyle='--', alpha=0.5)  # Slice boundary
        if perturbation[i] == 0:  # If the segment is "off" in the perturbation
            plt.axvspan(start_idx, end_idx, color='red', alpha=0.3)  # Highlight deactivated segment
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Plot perturbed ECG signal with slices and deactivated segments highlighted
    plt.subplot(2, 1, 2)
    plt.plot(perturbed_ecg, label='Perturbed ECG Signal', color='green')
    plt.title(f'Perturbed {title}')
    for i in range(num_slices):
        start_idx = i * slice_width
        end_idx = min((i + 1) * slice_width, len(original_ecg))
        plt.axvline(x=start_idx, color='r', linestyle='--', alpha=0.5)  # Slice boundary
        if perturbation[i] == 0:  # If the segment is "off" in the perturbation
            plt.axvspan(start_idx, end_idx, color='red', alpha=0.3)  # Highlight deactivated segment
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def plot_perturbed_ecg_all_channels(original_ecg, perturbed_ecg, perturbation, num_slices, title='ECG Signal with Perturbation'):
    """
    Plots the original and perturbed ECG signals with slices and deactivated segments highlighted.

    Parameters:
    - original_ecg (np.ndarray): The original ECG signal.
    - perturbed_ecg (np.ndarray): The perturbed ECG signal after applying the perturbation.
    - perturbation (np.ndarray): The perturbation vector used to modify the ECG signal.
    - num_slices (int): The total number of segments the ECG signal is divided into.
    - title (str): The title for the plot. Optional.
    """
    total_length = original_ecg.shape[1]
    slice_width = total_length // num_slices
    n_channels = original_ecg.shape[0]

    plt.figure(figsize=(12, 6))

    # Plot original ECG signal with slices and deactivated segments highlighted
    plt.subplot(2, 1, 1)
    for i in range(n_channels):
        plt.plot(original_ecg[i], color='tab:blue', alpha=0.5)
    plt.title(f'Original {title}')
    for i in range(num_slices):
        start_idx = i * slice_width
        end_idx = min((i + 1) * slice_width, total_length)
        plt.axvline(x=start_idx, color='r', linestyle='--', alpha=0.7)  # Slice boundary
        if perturbation[i] == 0:  # If the segment is "off" in the perturbation
            plt.axvspan(start_idx, end_idx, color='red', alpha=0.3)  # Highlight deactivated segment
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Plot perturbed ECG signal with slices and deactivated segments highlighted
    plt.subplot(2, 1, 2)
    for i in range(n_channels):
        plt.plot(perturbed_ecg[i], color='tab:green', alpha=0.7)
    plt.title(f'Perturbed {title}')
    for i in range(num_slices):
        start_idx = i * slice_width
        end_idx = min((i + 1) * slice_width, total_length)
        plt.axvline(x=start_idx, color='r', linestyle='--', alpha=0.5)  # Slice boundary
        if perturbation[i] == 0:  # If the segment is "off" in the perturbation
            plt.axvspan(start_idx, end_idx, color='red', alpha=0.3)  # Highlight deactivated segment
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def visualize_lime_explanation(instance_ecg, top_influential_segments, num_slices, perturb_function=perturb_mean, channel:int=0):
    """
    Visualizes the original ECG signal and highlights the top influential segments 
    identified by a LIME explanation.

    Parameters:
    - instance_ecg (np.ndarray): The original ECG signal.
    - top_influential_segments (np.ndarray): Indices of the top influential segments.
    - num_slices (int): The number of segments the ECG signal is divided into.
    - perturb_function (function): The perturbation function used (default is perturb_mean).
    """

    # grab specific channel
    instance_ecg_chan = instance_ecg[channel]

    # Initialize a mask with zeros
    n_samples = instance_ecg_chan.shape[-1]
    mask = np.zeros(n_samples)

    # Activate the top influential segments
    for segment in top_influential_segments:
        start_idx = segment * (n_samples // num_slices)
        end_idx = start_idx + (n_samples // num_slices)
        mask[start_idx:end_idx] = 1  # Set the segment indices to 1

    # Apply the mask to the original ECG signal
    perturbed_signal = apply_perturbation_to_ecg(instance_ecg, mask, num_slices, perturb_function)

    plt.figure(figsize=(12, 6))

    # Plot the original ECG signal
    plt.subplot(2, 1, 1)
    for i in range(1, num_slices):
        plt.axvline(x=i * (n_samples // num_slices), color='r', linestyle='--')
    plt.plot(instance_ecg_chan)
    for segment in top_influential_segments:
        start_idx = segment * (n_samples // num_slices)
        end_idx = start_idx + (n_samples // num_slices)
        plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)  # Highlight influential segments
    plt.title(f'Original Signal -- Channel {channel}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Plot the perturbed signal with highlighted segments
    plt.subplot(2, 1, 2)
    for i in range(1, num_slices):
        plt.axvline(x=i * (n_samples // num_slices), color='r', linestyle='--')
    plt.plot(perturbed_signal[channel], color='green')
    for segment in top_influential_segments:
        start_idx = segment * (n_samples // num_slices)
        end_idx = start_idx + (n_samples // num_slices)
        plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)  # Highlight influential segments
    plt.title('Perturbed Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def visualize_lime_explanation_all_channels(instance_ecg, top_influential_segments, num_slices, perturb_function=lime_explanation.perturb_mean):
    """
    Visualizes the original ECG signal and highlights the top influential segments 
    identified by a LIME explanation.

    Parameters:
    - instance_ecg (np.ndarray): The original ECG signal.
    - top_influential_segments (np.ndarray): Indices of the top influential segments.
    - num_slices (int): The number of segments the ECG signal is divided into.
    - perturb_function (function): The perturbation function used (default is perturb_mean).
    """
    # Initialize a mask with zeros
    n_samples = instance_ecg.shape[-1]
    n_channels = instance_ecg.shape[0]
    mask = np.zeros(n_samples)

    # Activate the top influential segments
    for segment in top_influential_segments:
        start_idx = segment * (n_samples // num_slices)
        end_idx = start_idx + (n_samples // num_slices)
        mask[start_idx:end_idx] = 1  # Set the segment indices to 1

    # Apply the mask to the original ECG signal
    perturbed_signal = lime_explanation.apply_perturbation_to_ecg(instance_ecg, mask, num_slices, perturb_function)

    plt.figure(figsize=(12, 6))

    # Plot the original ECG signal
    plt.subplot(2, 1, 1)
    for i in range(1, num_slices):
        plt.axvline(x=i * (n_samples // num_slices), color='r', linestyle='--')
    for chan in range(n_channels):
        plt.plot(np.arange(n_samples), instance_ecg[chan], alpha=0.5, color="tab:blue")
    for segment in top_influential_segments:
        start_idx = segment * (n_samples // num_slices)
        end_idx = start_idx + (n_samples // num_slices)
        plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)  # Highlight influential segments
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Plot the perturbed signal with highlighted segments
    plt.subplot(2, 1, 2)
    for i in range(1, num_slices):
        plt.axvline(x=i * (n_samples // num_slices), color='r', linestyle='--')
    for chan in range(n_channels):
        plt.plot(np.arange(n_samples), perturbed_signal[chan], color='tab:green', alpha=0.5)
    for segment in top_influential_segments:
        start_idx = segment * (n_samples // num_slices)
        end_idx = start_idx + (n_samples // num_slices)
        plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)  # Highlight influential segments
    plt.title('Perturbed Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def visualize_ave_lime_explanation(instance_ecg, top_influential_segments, num_slices, perturb_function=perturb_mean):
    """
    Visualizes the original ECG signal and highlights the top influential segments 
    identified by a LIME explanation.

    Parameters:
    - instance_ecg (np.ndarray): The original ECG signal.
    - top_influential_segments (np.ndarray): Indices of the top influential segments.
    - num_slices (int): The number of segments the ECG signal is divided into.
    - perturb_function (function): The perturbation function used (default is perturb_mean).
    """

    instance_ecg_chan = instance_ecg.mean(axis=0)

    # Initialize a mask with zeros
    n_samples = instance_ecg.shape[-1]
    mask = np.zeros(n_samples)

    # Activate the top influential segments
    for segment in top_influential_segments:
        start_idx = segment * (n_samples // num_slices)
        end_idx = start_idx + (n_samples // num_slices)
        mask[start_idx:end_idx] = 1  # Set the segment indices to 1

    # Apply the mask to the original ECG signal
    perturbed_signal = apply_perturbation_to_ecg(instance_ecg, mask, num_slices, perturb_function)

    plt.figure(figsize=(12, 6))

    # Plot the original ECG signal
    plt.subplot(2, 1, 1)
    for i in range(1, num_slices):
        plt.axvline(x=i * (n_samples // num_slices), color='r', linestyle='--')
    plt.plot(instance_ecg_chan, label='Original ECG Signal')
    for segment in top_influential_segments:
        start_idx = segment * (n_samples // num_slices)
        end_idx = start_idx + (n_samples // num_slices)
        plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)  # Highlight influential segments
    plt.title('Ave. Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Plot the perturbed signal with highlighted segments
    plt.subplot(2, 1, 2)
    for i in range(1, num_slices):
        plt.axvline(x=i * (n_samples // num_slices), color='r', linestyle='--')
    plt.plot(perturbed_signal.mean(axis=0), label='Highlighted ECG Signal', color='green')
    for segment in top_influential_segments:
        start_idx = segment * (n_samples // num_slices)
        end_idx = start_idx + (n_samples // num_slices)
        plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)  # Highlight influential segments
    plt.title('Ave. Perturbed Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
