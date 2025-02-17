from glob import glob
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
from itertools import groupby
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks
from pyentrp import entropy as ent

results = []


top_feats = ['autocorrelation', 'sampling_rate', 'event_density', 'trigger_ratio',
       'state_length', 'largest_state', 'dominant_state',
       '25th Percentile of Time Diffs (seconds)',
       '75th Percentile of Time Diffs (seconds)',
       'Median Absolute Deviation (MAD) of Time Diffs (seconds)',
       'Minimum Time Difference (seconds)', 'permutation_entropy',
       'sample_entropy', 'rms', 'energy', 'event_density_5T',
       'trigger_ratio_5T', 'state_length_5T', 'largest_state_5T',
       'dominant_state_5T', 'permutation_entropy_5T', 'rms_5T',
       'spectral_entropy_1H', 'event_density_1H', 'trigger_ratio_1H',
       'state_length_1H', 'largest_state_1H', 'dominant_state_1H',
       'permutation_entropy_1H', 'rms_1H', 'smallest_state_1D',
       'largest_state_1D', 'dominant_state_1D', 'rms_1D', 'event_density_1W',
       'trigger_ratio_1W', 'smallest_state_1W', 'largest_state_1W',
       'dominant_state_1W', 'rms_1W']


def determine_sampling_rate(time_series, timestamps):
    """
    Determines the sampling rate of a time series based on timestamps.

    Parameters:
        time_series (array-like): The time series data.
        timestamps (array-like): Corresponding timestamps for the time series.

    Returns:
        float: The calculated sampling rate.
    """
    time_deltas = np.diff(timestamps)
    return 1e9 / np.float32(np.mean(time_deltas))
def extract_time_series_features(time_series, sampling_rate=None, timestamps=None):
    """
    Extracts magnitude-independent features from a time series.

    Parameters:
        time_series (array-like): The time series data.
        sampling_rate (float or None): The sampling rate of the time series. If None, it will be determined from timestamps.
        timestamps (array-like or None): Corresponding timestamps for the time series, required if sampling_rate is None.

    Returns:
        dict: A dictionary containing the extracted features.
    """
    time_series = np.array(time_series)

    if sampling_rate is None:
        if timestamps is None:
            raise ValueError("Either sampling_rate or timestamps must be provided.")
        sampling_rate = determine_sampling_rate(time_series, timestamps)

    features = {}

    # Frequency Domain Features
    freqs, psd = welch(time_series, fs=sampling_rate)
    spectral_entropy = -np.sum(psd * np.log(psd + 1e-10))  # Avoid log(0)
    features['spectral_entropy'] = spectral_entropy
    features['dominant_frequency'] = freqs[np.argmax(psd)]

    # Shape-Based Features
    autocorr = np.correlate(time_series, time_series, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # Keep only non-negative lags
    features['autocorrelation'] = autocorr[1] if len(autocorr) > 1 else 0

    features['sampling_rate'] = sampling_rate

    return features

def extract_alarm_features(time_series):
    """
    Extract features specific to alarm time series.
    """
    features = {}
    binary_series = (time_series > 0).astype(int)
    features['event_density'] = np.mean(binary_series)
    features['trigger_ratio'] = np.sum(binary_series) / len(binary_series)
    return features

def extract_status_features(time_series):
    """
    Extract features specific to status time series.
    """
    features = {}
    unique_states, counts = np.unique(time_series, return_counts=True)
    features['state_length'] = len(unique_states)
    features['smallest_state'] = min(unique_states)
    features['largest_state'] = max(unique_states)
    features['dominant_state'] = unique_states[np.argmax(counts)]
    return features

def calculate_time_features(timestamps):
        # Convert timestamps to pandas datetime format if they are not already
        timestamps = pd.to_datetime(timestamps)

        # Sort timestamps to make sure the data is ordered chronologically
        timestamps = timestamps.sort_values()

        # Time differences between consecutive timestamps
        time_diffs = timestamps.diff().dropna()

        # Convert timedelta to total seconds for numerical operations
        time_diffs_seconds = time_diffs.dt.total_seconds()

        try:
            time_25th_percentile = np.percentile(time_diffs_seconds, 25)
        except:
            time_25th_percentile = 0
        try:
            time_75th_percentile = np.percentile(time_diffs_seconds, 75)
        except:
            time_75th_percentile = 0

        # Feature 10: Median absolute deviation (MAD) of time differences (in seconds)
        mad_time_diff = np.median(np.abs(time_diffs_seconds - np.median(time_diffs_seconds)))

        # Feature 13: Minimum and maximum time differences
        min_time_diff = time_diffs_seconds.min()


        # Organize features into a dictionary
        features = {
            "25th Percentile of Time Diffs (seconds)": time_25th_percentile,
            "75th Percentile of Time Diffs (seconds)": time_75th_percentile,
            "Median Absolute Deviation (MAD) of Time Diffs (seconds)": mad_time_diff,
            "Minimum Time Difference (seconds)": min_time_diff,
        }

        return features



def extract_entropy_features(time_series):
    """
    Extracts complexity and entropy-based features.
    """
    features = {}

    # Permutation Entropy
    try:
        perm_entropy = ent.permutation_entropy(time_series, order=3, delay=1)
        features['permutation_entropy'] = perm_entropy
    except:
        features['permutation_entropy'] = 0

    # Sample Entropy
    try:
        sample_entropy = ent.sample_entropy(time_series, sample_length=1, tolerance=0.2)[0]
        features['sample_entropy'] = sample_entropy
    except:
        features['sample_entropy'] = sample_entropy

    return features

def extract_time_domain_features(time_series):
    """
    Extracts additional time-domain features.
    """
    features = {}

    # Root Mean Square (RMS)
    try:
        rms = np.sqrt(np.mean(np.square(time_series)))
        features['rms'] = rms
    except:
        features['rms'] = 0


    # Signal Energy
    try:
        energy = np.sum(np.square(time_series))
        features['energy'] = energy
    except:
        features['energy'] = 0

    return features


def extract_features(config, df):
    if config == "interval_5T":
        df['time_bin'] = df['time'].dt.floor('5T')
    if config == "interval_1H":
        df['time_bin'] = df['time'].dt.floor('1H')
    if config == "interval_1D":
        df['time_bin'] = df['time'].dt.floor('1D')
    if config == "interval_1W":
        df['time_bin'] = df['time'].dt.floor('7D')

    if 'interval' in config:
        # Group by the 5-minute bins and calculate the mean
        df = df.groupby('time_bin', as_index=False)['value'].mean()
        df.columns = ["time", "value"]

    # General Features
    general_features = extract_time_series_features(time_series=df["value"].values,
                                                    timestamps=df["time"].to_numpy())
    # Specific Features
    alarm_features = extract_alarm_features(df["value"].values)
    status_features = extract_status_features(df["value"].values)

    # timestamp features
    timestamp_features = calculate_time_features(df["time"])

    # Add new features
    entropy_features = extract_entropy_features(df["value"].values)
    time_domain_features = extract_time_domain_features(df["value"].values)

    features = {**general_features, **alarm_features, **status_features, **timestamp_features,
                **entropy_features, **time_domain_features}

    if '_' in config:
        timeinterval = config.split('_')[-1]
        features = {f'{k}_{timeinterval}': v for k, v in features.items()}
    return features