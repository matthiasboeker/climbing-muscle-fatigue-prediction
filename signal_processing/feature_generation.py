from typing import Callable, Dict, List, Type
import numpy as np
import pandas as pd
from scipy.integrate import trapz
from scipy.signal import welch
from statsmodels.regression.linear_model import yule_walker
import neurokit2 as nk
import matplotlib.pyplot as plt

class SignalFeatures:
    def __init__(self, feature_name, sampling_rate, window_length, overlap):
        self.functions = {"emg": self.extract_emg_features,
                          "ppg": self.extract_ppg_features,
                          "imu": self.extract_imu_features,
                          "ecg": self.extract_ecg_features, 
                          "trajectory": self.extract_trajectory_features}
        self.feature_name = feature_name
        self.sampling_rate= sampling_rate 
        self.window_length = window_length 
        self.overlap = overlap

    @classmethod
    def set_params(cls, feature_name, sampling_rate, window_length, overlap):
        return cls(feature_name, sampling_rate, window_length, overlap)
    
    def get_features(self, signal):
        return sliding_window_features(signal, self.functions[self.feature_name], self.sampling_rate, self.window_length, self.overlap)

    #def get_annotations(self, annotations, signal_name):
    #    return sliding_window_annotations(annotations, self.functions[signal_name], self.sampling_rate, self.window_length, self.overlap)

    def extract_emg_features(self, window, fs) -> Dict[str, float]:
        window = window.emg
        mean = np.mean(window)
        mav = np.mean(np.abs(window))
        wl = np.sum(np.abs(np.diff(window)))
        iemg = trapz(np.abs(window))

        zc = len(np.where(np.diff(np.sign(window)))[0])

        ssc = len(np.where(np.diff(np.sign(np.diff(window))))[0])

        f, Pxx = welch(window, fs=fs, nperseg=len(window))

        mnf = np.sum(Pxx * f) / np.sum(Pxx)

        rms = np.sqrt(np.mean(np.square(window)))

        mdf = f[np.where(np.cumsum(Pxx) >= np.sum(Pxx) / 2)[0][0]]

        pkf = f[np.argmax(Pxx)]

        psd = np.sum(Pxx)

        rho, _ = yule_walker(window, order=1)
        features = {
                "MEAN": mean,
                'MAV': mav,
                'WL': wl,
                'IEMG': iemg,
                'ZC': zc,
                'SSC': ssc,
                'MNF': mnf,
                'RMS': rms,
                'MDF': mdf,
                'PKF': pkf,
                'PSD': psd,
                'Yule_Walker_Rho': rho[0],
            }

        return features

    def extract_imu_features(self, window, fs) -> Dict[str, float]:
        features = {
                'acc_x_mean': np.mean(window['acc_x']),
                'acc_x_std': np.std(window['acc_x']),
                'acc_y_mean': np.mean(window['acc_y']),
                'acc_y_std': np.std(window['acc_y']),
                'acc_z_mean': np.mean(window['acc_z']),
                'acc_z_std': np.std(window['acc_z']),
                'activity': np.sqrt(np.mean(window['acc_x'])**2 + np.mean(window['acc_y'])**2 + np.mean(window['acc_z'])**2),
            }
        return features
    
    def extract_trajectory_features(self, window, fs) -> Dict[str, float]:
        features = {
                'x': np.mean(window['COG_X']),
                'y': np.mean(window['COG_Y']),
            }
        return features

    def extract_ecg_features(self, signal, fs) -> Dict[str, float]:
        hr_rate = signal["ECG_Rate"]
        r_peaks = signal["ECG_R_Peaks"]
        rr_intervals = r_peaks.diff().dropna() / fs * 1000  # Convert to ms
        #hr_mean = 60000 / rr_intervals.mean()
        features = {
                'HR_mean': hr_rate.mean(),
                'HRV_SDNN': np.std(rr_intervals),
                'HRV_RMSSD': np.sqrt(np.mean(np.square(np.diff(rr_intervals)))),
            }
        return features

    def extract_ppg_features(self, signal, fs) -> Dict[str, float]:
        ppg_signals, _ = nk.ppg_process(signal, sampling_rate=fs)
        r_peaks = nk.ppg_findpeaks(ppg_signals['PPG_Raw'], sampling_rate=fs)
        rr_intervals = pd.Series(ppg_signals['PPG_Raw'][r_peaks['PPG_Peaks']]).diff().dropna() / fs * 1000  # Convert to ms
        hr_mean = 60000 / rr_intervals.mean()
        features = {
                'PPG_RR_mean':hr_mean,
                'PPG_HRV_SDNN': np.std(rr_intervals),
                'PPG_HRV_RMSSD': np.sqrt(np.mean(np.square(np.diff(rr_intervals)))),
            }
        return features
    
def sliding_window_features(signal, feature_func: Callable,  sampling_rate, window_length, overlap):
    if window_length < 5:
        raise ValueError("Window length must be at least 5 samples.")
        
    # Calculate step size based on overlap
    step_size = window_length - overlap
    num_windows = 1 + (len(signal) - window_length) // step_size
    if num_windows < 1:
        raise ValueError("Number of windows too low")
    if len(signal) < window_length:
        raise ValueError("Window length is larger than signal")
            
    features = []
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_length
        window = signal.iloc[start_idx:end_idx,:]
        features_window = feature_func(window, sampling_rate)
        features.append(features_window)
    return pd.DataFrame(features)
