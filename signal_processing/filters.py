
from typing import Any, Dict
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, lfilter
import pandas as pd
import neurokit2 as nk

def notch_filter(signal, notch_freq, Q, fs):
    """
    Apply a notch filter to the signal.

    Parameters:
    - signal: numpy array, the signal to be filtered.
    - notch_freq: float, the frequency to be removed (in Hz).
    - Q: float, quality factor.
    - fs: float, sampling frequency of the signal (in Hz).
    """
    # Design the notch filter
    b, a = iirnotch(notch_freq, Q, fs)

    # Apply the filter to the signal
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal

def butter_bandpass(lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

def butter_highpass(cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high')
        return b, a

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)  # filtfilt ensures zero phase distortion
    return y


class Preprocessor:
    def __init__(self, signal_preprocessing: Dict[str, Dict[str, Any]]):
        self.signal_preprocessing = signal_preprocessing

    def preprocess(self, signal_type: str, signal: Any) -> Any:
        if signal_type in self.signal_preprocessing:
            preprocess_method_name = f"preprocess_{signal_type}"
            if hasattr(self, preprocess_method_name):
                preprocess_method = getattr(self, preprocess_method_name)
                kwargs = self.signal_preprocessing[signal_type]
                return preprocess_method(signal, **kwargs)
        return signal
    

    def preprocess_emg(self, sensor_recording: Any, **kwargs) -> Any:
        signal = sensor_recording.signal.emg
        b, a = butter_bandpass(kwargs.get('emg_lowcut', 20), kwargs.get('emg_highcut', 450), kwargs.get('emg_fs', 2000), order=kwargs.get('emg_order', 4))
        filtered_signal = filtfilt(b, a, signal)
        rectified_signal = np.abs(filtered_signal)
        
        b, a = butter_lowpass(10, kwargs.get('emg_fs', 2000), order=kwargs.get('emg_order', 4))
        smoothed_signal = filtfilt(b, a, rectified_signal)
        signal.emg = smoothed_signal
        sensor_recording.signal = signal
        return sensor_recording

    def preprocess_ecg(self, sensor_recording, **kwargs) -> Any:
        signal = sensor_recording.signal.ecg
        signals, _ = nk.ecg_process(signal.values, kwargs.get('ecg_sampling_rate', 500))
        signal["ecg"] = signals["ECG_Clean"]
        sensor_recording.signal = signal
        return sensor_recording

    def preprocess_imu(self, sensor_recording: pd.DataFrame, **kwargs) -> pd.DataFrame:
        signal = sensor_recording.signal
        b, a = butter_lowpass(kwargs.get('imu_cutoff',0.1), kwargs.get('imu_fs', 50), order=kwargs.get('imu_order', 4))
        signal['acc_x'] = filtfilt(b, a, signal['acc_x'])
        signal['acc_y'] = filtfilt(b, a, signal['acc_y'])
        signal['acc_z'] = filtfilt(b, a, signal['acc_z'])
        signal['w'] = filtfilt(b, a, signal['w'])
        signal['x'] = filtfilt(b, a, signal['x'])
        signal['y'] = filtfilt(b, a, signal['y'])
        signal['z'] = filtfilt(b, a, signal['z'])
        sensor_recording.signal = signal
        return sensor_recording

    def preprocess_ppg(self, sensor_recording: Any, **kwargs) -> Any:
        signal = sensor_recording.signal
        for channel in ["g", "b", "r", "ir"]:
            signals, info = nk.ppg_process(signal[channel], kwargs.get("ppg_sampling_rate", 50))
            signal[channel] = signals["PPG_Clean"]
        sensor_recording.signal = signal
        return sensor_recording
    
    def preprocess_eda(self, sensor_recording: Any, **kwargs) -> Any:
        signal = sensor_recording.signal
        signals, info = nk.eda_process(signal["eda"], kwargs.get("eda_sampling_rate", 500))
        signal["eda"] = signals["EDA_Clean"]
        sensor_recording.signal = signal
        return sensor_recording
    