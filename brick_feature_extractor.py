from glob import glob
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
from itertools import groupby
from scipy.stats import skew, kurtosis
from scipy.signal import welch, find_peaks

results = []


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

    # Statistical Moments
    features['skewness'] = skew(time_series)
    features['kurtosis'] = kurtosis(time_series)

    # Frequency Domain Features
    freqs, psd = welch(time_series, fs=sampling_rate)
    spectral_entropy = -np.sum(psd * np.log(psd + 1e-10))  # Avoid log(0)
    features['spectral_entropy'] = spectral_entropy
    features['dominant_frequency'] = freqs[np.argmax(psd)]

    # Shape-Based Features
    autocorr = np.correlate(time_series, time_series, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # Keep only non-negative lags
    features['autocorrelation'] = autocorr[1] if len(autocorr) > 1 else 0

    peaks, _ = find_peaks(time_series)
    valleys, _ = find_peaks(-time_series)
    features['peak_count'] = len(peaks)
    features['valley_count'] = len(valleys)

    # Temporal Patterns
    time_series_diff = np.diff(time_series)
    features['trend'] = np.mean(time_series_diff)
    seasonal_variation = np.std(time_series - np.mean(time_series))
    features['seasonal_variation'] = seasonal_variation

    features['sampling_rate'] = sampling_rate

    return features

def extract_sensor_features(time_series):
    """
    Extract features specific to sensor time series.
    """
    features = {}
    features['noise_level'] = np.std(time_series)
    features['dynamic_range'] = np.ptp(time_series)  # Peak-to-peak range
    features['periodic_power'] = np.sum(np.abs(np.fft.rfft(time_series)[1:]))  # Ignore DC component
    return features

def extract_alarm_features(time_series):
    """
    Extract features specific to alarm time series.
    """
    features = {}
    binary_series = (time_series > 0).astype(int)
    features['event_density'] = np.mean(binary_series)
    features['event_duration'] = np.mean(np.diff(np.where(binary_series == 1)[0])) if np.any(binary_series) else 0
    features['event_intervals'] = np.std(np.diff(np.where(binary_series == 1)[0])) if np.any(binary_series) else 0
    features['trigger_ratio'] = np.sum(binary_series) / len(binary_series)
    return features

def extract_setpoint_features(time_series):
    """
    Extract features specific to setpoint time series.
    """
    features = {}
    changes = np.diff(time_series)
    try:
        features['change_frequency'] = np.sum(changes != 0)
    except:
        features['change_frequency'] = 0
    try:
        features['step_size_mean'] = np.mean(changes[np.nonzero(changes)]) if np.any(changes) else 0
    except:
        features['step_size_mean'] = 0
    try:
        features['step_size_std'] = np.std(changes[np.nonzero(changes)]) if np.any(changes) else 0
    except:
        features['step_size_std'] = 0
    try:
        features['plateau_duration'] = np.mean([len(list(group)) for key, group in groupby(time_series)])
    except:
        features['plateau_duration'] = 0
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

def extract_parameter_features(time_series):
    """
    Extract features specific to parameter time series.
    """
    features = {}
    features['value_stability'] = np.std(time_series)
    features['slope_variability'] = np.std(np.diff(time_series))
    features['outlier_ratio'] = np.sum(np.abs(time_series - np.mean(time_series)) > 2 * np.std(time_series)) / len(time_series)
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

        # Feature 1: Average time between samples (in seconds)
        avg_time_between_samples = time_diffs_seconds.mean()

        # Feature 2: Time variance (variance in the time difference)
        time_variance = time_diffs_seconds.var()

        # Feature 3: Time standard deviation (in seconds)
        time_std_dev = time_diffs_seconds.std()

        # Feature 4: Time range (difference between max and min timestamp)
        time_range = (timestamps.max() - timestamps.min()).total_seconds()

        # Feature 5: Median time between samples (in seconds)
        median_time_between_samples = time_diffs_seconds.median()

        # Feature 6: 25th and 75th percentiles (interquartile range of time diffs, in seconds)
        try:
            time_25th_percentile = np.percentile(time_diffs_seconds, 25)
        except:
            time_25th_percentile = 0
        try:
            time_75th_percentile = np.percentile(time_diffs_seconds, 75)
        except:
            time_75th_percentile = 0

        # Feature 7: Time skewness (degree of asymmetry in time differences)
        time_skewness = pd.Series(time_diffs_seconds).skew()

        # Feature 8: Time kurtosis (measure of the "tailedness" of the time difference distribution)
        time_kurtosis = pd.Series(time_diffs_seconds).kurtosis()

        # Feature 9: Count of samples (number of timestamps)
        sample_count = len(timestamps)

        # Feature 10: Median absolute deviation (MAD) of time differences (in seconds)
        mad_time_diff = np.median(np.abs(time_diffs_seconds - np.median(time_diffs_seconds)))

        # Feature 11: Time mode (most frequent time difference)
        time_mode = time_diffs_seconds.mode()[0] if not time_diffs_seconds.empty else 0

        # Feature 12: Inter-arrival time (mean time between non-zero intervals)
        non_zero_intervals = time_diffs_seconds[time_diffs_seconds > 0]
        avg_non_zero_intervals = non_zero_intervals.mean() if not non_zero_intervals.empty else 0

        # Feature 13: Minimum and maximum time differences
        min_time_diff = time_diffs_seconds.min()
        max_time_diff = time_diffs_seconds.max()

        # Feature 14: Time periodicity (variance in time differences)
        time_periodicity = np.var(time_diffs_seconds)

        # Feature 15: Time autocorrelation at lag 1
        time_autocorr = np.corrcoef(time_diffs_seconds[:-1], time_diffs_seconds[1:])[0, 1] if len(
            time_diffs_seconds) > 1 else 0

        # Feature 16: Time trend slope (linear fit slope of time series)
        try:
            time_slope = np.polyfit(np.arange(len(timestamps)-1), time_diffs_seconds, 1)[0]
        except:
            time_slope=0

        # Feature 17: Number of significant time gaps (greater than a threshold)
        try:
            gap_threshold = np.percentile(time_diffs_seconds, 95)  # 95th percentile as a threshold for gaps
        except:
            gap_threshold = 0

        try:
            significant_gaps = np.sum(time_diffs_seconds > gap_threshold)
        except:
            significant_gaps = 0

        # Feature 18: Coefficient of variation of time differences
        try:
            coeff_of_variation = time_std_dev / avg_time_between_samples if avg_time_between_samples != 0 else 0
        except:
            coeff_of_variation = 0
        # Feature 19: Skewness of the cumulative sum of time differences
        try:
            cum_sum_skewness = pd.Series(np.cumsum(time_diffs_seconds)).skew()
        except:
            cum_sum_skewness = 0

        # Organize features into a dictionary
        features = {
            "Average Time Between Samples (seconds)": avg_time_between_samples,
            "Time Variance (seconds^2)": time_variance,
            "Time Standard Deviation (seconds)": time_std_dev,
            "Time Range (seconds)": time_range,
            "Median Time Between Samples (seconds)": median_time_between_samples,
            "25th Percentile of Time Diffs (seconds)": time_25th_percentile,
            "75th Percentile of Time Diffs (seconds)": time_75th_percentile,
            "Time Skewness": time_skewness,
            "Time Kurtosis": time_kurtosis,
            "Sample Count": sample_count,
            "Median Absolute Deviation (MAD) of Time Diffs (seconds)": mad_time_diff,
            "Time Mode (seconds)": time_mode,
            "Average Non-Zero Inter-Arrival Time (seconds)": avg_non_zero_intervals,
            "Minimum Time Difference (seconds)": min_time_diff,
            "Maximum Time Difference (seconds)": max_time_diff,
            "Time Periodicity (variance)": time_periodicity,
            "Time Autocorrelation (lag 1)": time_autocorr,
            "Time Trend Slope": time_slope,
            "Number of Significant Time Gaps": significant_gaps,
            "Coefficient of Variation of Time Diffs": coeff_of_variation,
            "Skewness of Cumulative Sum of Time Diffs": cum_sum_skewness
        }

        return features





def extract_fractal_dimension(time_series):
    """
    Extracts the fractal dimension of the time series.
    """
    features = {}

    try:
        # Approximation of fractal dimension using Higuchi's method
        def higuchi_fd(time_series, kmax=10):
            n = len(time_series)
            L = np.zeros(kmax)
            for k in range(1, kmax + 1):
                L[k - 1] = np.sum(np.abs(time_series[k:] - time_series[:-k])) / k
            return np.log(L[-1] / L[0]) / np.log(kmax)

        fractal_dimension = higuchi_fd(time_series)
        features['fractal_dimension'] = fractal_dimension
    except:
        features['fractal_dimension'] = 0

    return features

def extract_event_based_features(time_series):
    """
    Extracts event-based features like burstiness and inter-event time features.
    """
    features = {}
    try:
        # Burstiness
        binary_series = (time_series > np.mean(time_series)).astype(int)  # Simple thresholding to detect events
        event_indices = np.where(binary_series == 1)[0]
        burstiness = len(event_indices) / len(time_series)
        features['burstiness'] = burstiness

        # Inter-Event Time Features
        if len(event_indices) > 1:
            inter_event_times = np.diff(event_indices)
            features['mean_inter_event_time'] = np.mean(inter_event_times)
            features['std_inter_event_time'] = np.std(inter_event_times)
            features['max_inter_event_time'] = np.max(inter_event_times)
        else:
            features['mean_inter_event_time'] = 0
            features['std_inter_event_time'] = 0
            features['max_inter_event_time'] = 0
    except:
        features['burstiness'] = 0
        features['mean_inter_event_time'] = 0
        features['std_inter_event_time'] = 0
        features['max_inter_event_time'] = 0

    return features

from pyentrp import entropy as ent

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

    # Hurst Exponent
    try:
        hurst_exponent = np.polyfit(np.log(np.arange(1, len(time_series))), np.log(np.cumsum(np.abs(np.diff(time_series)))), 1)[0]
        features['hurst_exponent'] = hurst_exponent
    except:
        features['hurst_exponent'] = 0

    return features

def extract_frequency_domain_features(time_series, sampling_rate):
    """
    Extracts frequency-domain features.
    """
    features = {}
    try:
        # Frequency Domain Features (from previous code)
        freqs, psd = welch(time_series, fs=sampling_rate)
        spectral_entropy = -np.sum(psd * np.log(psd + 1e-10))
        features['spectral_entropy'] = spectral_entropy
        features['dominant_frequency'] = freqs[np.argmax(psd)]

        # Band Energy Ratios (example bands: low < 5Hz, mid 5-50Hz, high > 50Hz)
        low_band = np.sum(psd[freqs < 5])
        mid_band = np.sum(psd[(freqs >= 5) & (freqs <= 50)])
        high_band = np.sum(psd[freqs > 50])
        total_energy = low_band + mid_band + high_band
        features['low_band_energy_ratio'] = low_band / total_energy
        features['mid_band_energy_ratio'] = mid_band / total_energy
        features['high_band_energy_ratio'] = high_band / total_energy

        # Spectral Centroid
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        features['spectral_centroid'] = spectral_centroid

        # Spectral Flatness
        spectral_flatness = np.exp(np.mean(np.log(psd + 1e-10))) / np.mean(psd)
        features['spectral_flatness'] = spectral_flatness
    except:
        features['spectral_entropy'] = 0
        features['dominant_frequency'] = 0
        features['low_band_energy_ratio'] = 0
        features['mid_band_energy_ratio'] = 0
        features['high_band_energy_ratio'] = 0
        features['spectral_centroid'] = 0
        features['spectral_flatness'] = 0
    return features

def extract_time_domain_features(time_series):
    """
    Extracts additional time-domain features.
    """
    features = {}

    # Zero-Crossing Rate
    try:
        zero_crossings = np.count_nonzero(np.diff(np.sign(time_series)))
        features['zero_crossing_rate'] = zero_crossings / len(time_series)
    except:
        features['zero_crossing_rate'] = 0

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

    # Coefficient of Variation
    try:
        mean_val = np.mean(time_series)
        std_val = np.std(time_series)
        features['coefficient_of_variation'] = std_val / mean_val if mean_val != 0 else 0
    except:
        features['coefficient_of_variation'] = 0

    return features


def time_interval_features(df):
    ## # Calculate time differences in seconds
    df['time_diff'] = df['time'].diff().dt.total_seconds()
    ## # Find the most common interval (mode)
    expected_interval = df['time_diff'].mode()[0]  # Mode of time differences
    ## # Identify deviations
    deviations = df['time_diff'] != expected_interval
    ## # Count the number of deviations (excluding the first row due to NaN)
    deviation_count = deviations.sum()
    ##
    return {"exp_interval":expected_interval, 'outside_interval':deviation_count}
    # print(f"Inferred expected interval: {expected_interval} seconds")
    # print(f"Number of timestamps that do not follow the expected interval: {deviation_count}")


####### MAIN #########

config = 'FULL'
mode = "TRAIN"
lf = pd.read_csv('train_y_v0.1.0.csv')
all_features = []
for file in tqdm(glob("train_x/*.pkl")):#tqdm(glob("/Users/bsteenwi/Downloads/test_X/*.pkl")): ##
    with open(file, 'rb') as f:
        name = file.split('/')[-1]
        time_series_data = pickle.load(f)
        df = pd.DataFrame(time_series_data)
        df.columns = ["time", "value"]
        start_date = pd.Timestamp('2023-01-01')
        df['time'] = start_date + df['time']

        if config == "time":
            feats = time_interval_features(df)
            feats['filename'] = name
            all_features.append({**feats})

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
        sensor_features = extract_sensor_features(df["value"].values)
        alarm_features = extract_alarm_features(df["value"].values)
        setpoint_features = extract_setpoint_features(df["value"].values)
        status_features = extract_status_features(df["value"].values)
        parameter_features = extract_parameter_features(df["value"].values)

        # timestamp features
        timestamp_features = calculate_time_features(df["time"])

        # Add new features
        fractal_features = extract_fractal_dimension(df["value"].values)
        event_based_features = extract_event_based_features(df["value"].values)
        entropy_features = extract_entropy_features(df["value"].values)
        freq_features = extract_frequency_domain_features(df["value"].values, general_features["sampling_rate"])
        time_domain_features = extract_time_domain_features(df["value"].values)

        # Combine all features
        all_features.append(
            {**general_features, **sensor_features, **alarm_features, **setpoint_features, **status_features,
             **parameter_features, **timestamp_features, **fractal_features, **event_based_features,
             **entropy_features, **freq_features, **time_domain_features, 'filename': name})


df = pd.DataFrame(all_features)
if mode == "TRAIN":
    lf = pd.read_csv('train_y_v0.1.0.csv')
    final_df = df.merge(lf, on='filename')
    final_df.to_csv(f'extended_train_{config}_features.csv')
else:
    df.to_csv(f'extended_test_{config}_features.csv')