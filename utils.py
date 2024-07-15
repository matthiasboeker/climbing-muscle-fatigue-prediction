import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import hilbert, butter, filtfilt
from scipy.spatial.distance import euclidean
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from scipy.integrate import cumtrapz
from fastdtw import fastdtw
from scipy.stats import linregress
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot


def acf_pacf_plots(df, features):
    fig, axes = plt.subplots(len(features), 2, figsize=(15, 15))

    # Loop through features to create the ACF and PACF plots
    for i, feature in enumerate(features):
        # Plot ACF
        plot_acf(df[feature], ax=axes[i, 0], lags=20, title=f'ACF of {feature}')
        # Plot PACF
        plot_pacf(df[feature], ax=axes[i, 1], lags=20, title=f'PACF of {feature}')

    plt.tight_layout()
    plt.show()

# Function to plot autocorrelation
def plot_autocorrelation(ax, series, lags=400):
    autocorr = acf(series, nlags=lags, fft=True)
    ax.bar(range(lags + 1), autocorr)
    ax.set_ylim([0, 1])

# Function to plot cross-correlation
def plot_cross_correlation(ax, series1, series2, lags=400):
    crosscorr = ccf(series1, series2)[:2*lags+1]
    crosscorr = np.concatenate([crosscorr[lags:], crosscorr[:lags][::-1]])
    ax.bar(range(-lags, lags + 1), crosscorr)
    ax.set_ylim([-0.25, 0.5])


def plot_ts_correlation(df, features):
    plt.rcParams.update({
        'font.size': 24,      # Sets the base font size
        'axes.titlesize': 26, # Sets the subplot title font size
        'axes.labelsize': 24, # Sets the x and y labels font size
        'xtick.labelsize': 22, # Sets the x tick labels font size
        'ytick.labelsize': 22, # Sets the y tick labels font size
    })
    # Create subplots
    fig, axes = plt.subplots(len(features), len(features), figsize=(25, 25))

    # Loop through features to create the grid of plots
    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):
            ax = axes[i, j]
            if i == j:
                # Autocorrelation plot on the diagonal
                plot_autocorrelation(ax, df[feature_i])
            else:
                # Cross-correlation plot on the off-diagonal
                plot_cross_correlation(ax, df[feature_i], df[feature_j])
            if j == 0:
                ax.set_ylabel(feature_i)
            if i == len(features) - 1:
                ax.set_xlabel(feature_j)

    plt.tight_layout()
    plt.savefig("cc_plot.jpeg")

def plot_trajectory_density(df):
    reversed_y = reverse_y_axis(df["y"])  # Reverse y-values
    df_new = df.copy()
    df_new["reversed_y"] = reversed_y
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df_new, x='x', y='reversed_y', fill=True, cmap='viridis', cbar=True)
    plt.title('KDE of Climber Trajectories on X and Y Plane')
    plt.xlabel('X Position')
    plt.xlim(0.2, 1.75)
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.show()

def plot_density_fatigue_height(df):
    df = df[df['y'] >= 0]

    df["reversed_y"] = reverse_y_axis(df["y"])
    df['height_bin'] = np.floor(df["reversed_y"])
    
    # Define the range of MNF values for plotting
    min_mnf = df['MNF'].min()
    max_mnf = df['MNF'].max()
    mnf_range = np.linspace(min_mnf, max_mnf, 300)

    fig, ax = plt.subplots(figsize=(25, 10))
    # Get the unique heights and normalize them for color mapping
    unique_heights = sorted(df['height_bin'].unique(), reverse=True)
    norm = Normalize(vmin=min(unique_heights), vmax=max(unique_heights))
    cmap = plt.get_cmap('viridis')  # You can change 'viridis' to any other gradient colormap

    # Lists to store median points
    median_heights = []
    median_values = []
    for height in unique_heights:
        subset = df[df['height_bin'] == height]['MNF']
        if not subset.empty:
            kde = gaussian_kde(subset)
            density = kde(mnf_range)
            density /= density.max()  # Normalize for visual consistency
            cumulative_density = cumtrapz(density, mnf_range, initial=0)
            cumulative_density /= cumulative_density[-1]

            # Find the median
            median_index = np.abs(cumulative_density - 0.5).argmin()
            median_mnf = mnf_range[median_index]
            median_heights.append(height)
            median_values.append(median_mnf)

            # Retrieve color from the colormap
            color = cmap(norm(height))

            ax.fill_betweenx(mnf_range, height - density, height, color=color)
            ax.plot(height, median_mnf, 'ko')  # Plot the median point

    # Plot the line connecting the medians
    ax.plot(median_heights, median_values, 'o--', color='darkgrey', markersize=5, linewidth=1.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_facecolor('#f4f4f4')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax.set_title('Density of EMG Median Frequency Across Different Heights', fontsize=28)
    ax.set_xlabel('Height (meters)', fontsize=26)
    ax.set_ylabel('MNF', fontsize=26)
    plt.show()

def plot_muscle_contraction_trajectory(envelope_pred, x_trajectory, y_trajectory):
    reversed_y = reverse_y_axis(y_trajectory)  # Reverse y-values
    contraction_intensity = envelope_pred / np.max(envelope_pred)  # Normalize contraction intensity
    
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(x_trajectory, reversed_y, c=contraction_intensity, cmap='hot', s=120, edgecolor='k', alpha=0.7)
    
    # Adding a color bar with a better label
    cbar = plt.colorbar(sc)
    cbar.set_label('Muscle Contraction Intensity', rotation=270, labelpad=20, fontsize=20)
    
    # Adding grid lines for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Enhancing labels and title
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)
    plt.title("Climber's Trajectory with Muscle Contractions", fontsize=22)
    
    plt.show()

def reverse_y_axis(y_values):
    return [max(y_values) - y for y in y_values]

def plot_fatigue_trajectory(fatigue, x_trajectory, y_trajectory):
    reversed_y = reverse_y_axis(y_trajectory)
    slope, intercept, r_value, p_value, std_err = linregress(reversed_y, fatigue)
    print()

    # Calculate the predicted fatigue values for color coding
    fatigue_predicted = [slope * y + intercept for y in reversed_y]
    # Calculate min and max of the predicted fatigue values
    min_fatigue = min(fatigue_predicted)
    max_fatigue = max(fatigue_predicted)

    # Convert predicted fatigue to percentage of the range
    fatigue_percent = [(f - min_fatigue) / (max_fatigue - min_fatigue) * 100 for f in fatigue_predicted]


    plt.figure(figsize=(12, 10))
    # Scatter plot with improved aesthetics, fixed color range, and reversed colormap
    sc = plt.scatter(x_trajectory, reversed_y, c=fatigue_percent, cmap='hot', s=120, edgecolor='k', alpha=0.7, vmin=5, vmax=100)
    
    # Adding a color bar with a better label
    cbar = plt.colorbar(sc)
    cbar.set_label('Muscle Energy', rotation=270, labelpad=20, fontsize=20)
    
    # Adding grid lines for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Enhancing labels and title
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)
    plt.title("Climber's Trajectory with Muscle Fatigue", fontsize=22)
    
    # Invert y-axis to keep positive labels
    plt.show()


def plot_muscle_contractions(emg_signal, envelope_true, envelope_pred):
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(len(emg_signal)), emg_signal, color='r', label='True Signal')
    plt.plot(np.arange(len(envelope_true)), envelope_true*0.8, color='g', label='Envelope')
    plt.plot(np.arange(len(envelope_pred)), envelope_pred, color='b', label='Pred Envelope')
    plt.xlabel('Time')
    plt.ylabel('MNF')
    plt.title(f"Muscle contractions)")
    plt.legend()
    plt.show()



def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(12, 9))  # Increased figure size for better readability
    plt.plot(np.arange(len(y_true)), y_true, 'r-', label='True Fatigue', linewidth=2)  # Thicker line
    plt.plot(np.arange(len(y_pred)), y_pred, 'b--', marker="o", label='Prediction of Fatigue', linewidth=2)  # Thicker line and dashed

    # Enhancing the tick marks for better visibility
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # Enlarging labels and title
    plt.xlabel('Time', fontsize=26)
    plt.ylabel('MNF', fontsize=26)
    plt.title('Fatigue Prediction', fontsize=18)

    # Enlarging the legend
    plt.legend(fontsize=24)

    # Adding grid for better readability
    plt.grid(True)

    plt.show()

def detect_muscle_contractions(iemg_signal, fs=4, cutoff=0.5, percentile=75):
    analytic_signal = hilbert(iemg_signal)
    envelope = np.abs(analytic_signal)
    
    def lowpass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
    envelope = lowpass_filter(envelope, cutoff, fs)
    
    # Calculate adaptive threshold
    threshold = np.percentile(envelope, percentile)

    # Generate binary signal
    binary_signal = (envelope > threshold).astype(int)

    # Create new signal with original values where binary_signal is true
    filtered_iemg_signal = iemg_signal * binary_signal

    return binary_signal, filtered_iemg_signal

# Hamming Loss
def hamming_loss(true_signal, predicted_signal):
    return np.mean(true_signal != predicted_signal)

# Accuracy
def accuracy(true_signal, predicted_signal):
    return np.mean(true_signal == predicted_signal)

# Intersection over Union (IoU)
def intersection_over_union(true_signal, predicted_signal):
    intersection = np.sum((true_signal == 1) & (predicted_signal == 1))
    union = np.sum((true_signal == 1) | (predicted_signal == 1))
    return intersection / union if union != 0 else 1

# DTW Loss
def dtw_loss(true_signal, predicted_signal):
    dtw_distance, _ = fastdtw(np.expand_dims(true_signal, 1), np.expand_dims(predicted_signal, 1), dist=euclidean)
    return dtw_distance

# Read in trajectories function
def read_in_trajectories(path_to_file: Path):
    return pd.read_csv(path_to_file)

def create_lagged_features(df, lags=3):
    lagged_data = {}
    for lag in range(1, lags + 1):
        lagged_data[f'xm{lag}'] = df['x'].shift(lag, fill_value=df['x'][0])
        lagged_data[f'acc_xm{lag}'] = df['acc_x_mean'].shift(lag, fill_value=df['acc_x_mean'][0])
        lagged_data[f'acc_ym{lag}'] = df['acc_y_mean'].shift(lag, fill_value=df['acc_y_mean'][0])
        lagged_data[f'ym{lag}'] = df['y'].shift(lag, fill_value=df['y'][0])
        lagged_data[f'mnfm{lag}'] = df['MNF'].shift(lag, fill_value=df['MNF'][0])
        lagged_data[f'hrm{lag}'] = df['HR_mean'].shift(lag, fill_value=df['HR_mean'][0])
    
    lagged_df = pd.DataFrame(lagged_data)
    return pd.concat([df, lagged_df], axis=1)

def plot_regularization_path(alphas, coefs, X):
    # Get the alphas and the corresponding coefficients
    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[0]):
        plt.plot(alphas, coefs[:, i], label=X.columns[i])

    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficients')
    plt.title('Regularization Path')
    plt.legend()
    plt.axis('tight')
    plt.show()

def plot_analytics_graphic(residuals):
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    fig = plot_acf(residuals, lags=40, ax=ax[0, 0])
    fig = plot_pacf(residuals, lags=40, ax=ax[0, 1])
    fig = qqplot(residuals, line='s', ax=ax[1, 0])
    ax[1, 1].plot(residuals)
    ax[1, 1].set_title('Residuals')
    plt.show()

def scale_dataframe(df, scaler, y_col):
    scaler.fit(df)

    df_scaled = scaler.transform(df)

    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    return df_scaled.drop([y_col], axis=1), df_scaled[y_col]

def scale_transform_dataframe(df, scaler, y_col):

    df_scaled = scaler.transform(df)

    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    return df_scaled.drop([y_col], axis=1), df_scaled[y_col]

def detrend_time_series(data):
    time = np.arange(len(data))    
    trend = np.polyfit(time, data, 1)
    trend_line = np.polyval(trend, time)
    
    detrended_data = data - trend_line    
    return detrended_data

def estimate_trend(data):
    time = np.arange(len(data))
    coefficients = np.polyfit(time, data, 1)
    return coefficients