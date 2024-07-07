from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import neurokit2 as nk

from data_structure.read_in_functions import get_signal_events, calculate_sliding_window_params, read_in_trajectories
from data_structure.structure_classes import Eventtype, Signaltype, Side, Labels, get_signal_, sliding_window_annotations
from signal_processing.filters import Preprocessor
from signal_processing.feature_generation import SignalFeatures

def pad_zeros(df, n):
    zeros_df = pd.DataFrame(0, index=range(n), columns=df.columns)
    return  pd.concat([df, zeros_df], axis=0)

signals_sampling_rates = {
    "emg": 2000,
    "ppg": 500,
    "imu": 50,
    "ecg": 500,
    "trajectory": 30
}

def process_and_save(participant_nr, sliding_window_params,  climbing_style, signal_events, trajectories, path_to_save):
    emg_feature_extractor = SignalFeatures("emg", signals_sampling_rates["emg"], sliding_window_params["emg"][0], sliding_window_params["emg"][1])
    imu_feature_extractor = SignalFeatures("imu", signals_sampling_rates["imu"], sliding_window_params["imu"][0], sliding_window_params["imu"][1])
    ecg_feature_extractor = SignalFeatures("ecg", signals_sampling_rates["ecg"], sliding_window_params["ecg"][0], sliding_window_params["ecg"][1])
    traj_feature_extractor = SignalFeatures("trajectory", signals_sampling_rates["trajectory"], sliding_window_params["trajectory"][0], sliding_window_params["trajectory"][1])
    
    
    emg_signal = get_signal_(signal_events, event_type=climbing_style, signal_type=Signaltype.emg, side=Side.right)[0]
    imu_signal = get_signal_(signal_events, event_type=climbing_style, signal_type=Signaltype.imu, side=Side.right)[0]
    ecg_signal = get_signal_(signal_events, event_type=climbing_style, signal_type=Signaltype.ecg, side=Side.right)[0]
    trajectory = [traj for traj in trajectories if traj["style"] == climbing_style][0]["trajectory"]
    annotations = pd.concat(sliding_window_annotations(emg_signal.signal, sampling_rate=signals_sampling_rates["emg"], window_length=sliding_window_params["emg"][0], overlap=sliding_window_params["emg"][1]), axis=1).T.reset_index(drop=True)
    r_peaks,_ = nk.ecg_process(ecg_signal.signal.ecg, sampling_rate=500)
    #print(emg_signal.signal[[f"quarter {i}" for i in range(1,5)]].sum(axis=0))
    print(r_peaks)
    features_ecg = ecg_feature_extractor.get_features(r_peaks)
    features_emg = emg_feature_extractor.get_features(emg_signal.signal)
    features_imu = imu_feature_extractor.get_features(imu_signal.signal)
    features_traj = traj_feature_extractor.get_features(trajectory)
    
    longest = max(len(features_emg), len(features_imu), len(features_traj))
    
    features_emg = pad_zeros(features_emg, longest - len(features_emg))
    features_imu = pad_zeros(features_imu, longest - len(features_imu))
    features_traj = pad_zeros(features_traj, longest - len(features_traj))
    features_ecg = pad_zeros(features_ecg, longest - len(features_ecg))
    # Reset index to ensure unique indices for concatenation
    features_emg.reset_index(drop=True, inplace=True)
    features_imu.reset_index(drop=True, inplace=True)
    features_ecg.reset_index(drop=True, inplace=True)
    features_traj.reset_index(drop=True, inplace=True)
    
    merged_features = pd.concat([features_emg, features_imu, features_ecg, features_traj, annotations], axis=1)
    merged_features.to_csv(path_to_save / f"{participant_nr}_{climbing_style}.csv", index=False)


def main():
   path_to_thermo = Path(__file__).parent / "data" / "anxiety_thermometer.csv"
   path_to_trajectories = Path(__file__).parent / "data" / "trajectories" / "trajectories_transformed"
   path_to_participants = Path(__file__).parent / "data" / "trial_1" 
   path_to_temp_folder = Path(__file__).parent / "data" / "temporary_files" 
   anxiety_thermometer = pd.read_csv(path_to_thermo, sep=";")
   sliding_window_params = calculate_sliding_window_params(signals_sampling_rates, 0.5, 0.25)
   print(sliding_window_params)

   for nr in [18]:
      print(f"Participant {nr}")
      trajectories = read_in_trajectories(path_to_trajectories, [nr])
      signal_events = get_signal_events(path_to_participants / f"participant_{nr}", nr)
      process_and_save(nr, sliding_window_params,Eventtype.lead_climbing.value, signal_events, trajectories, path_to_temp_folder)
      process_and_save(nr, sliding_window_params,Eventtype.toprope_climbing.value, signal_events, trajectories, path_to_temp_folder)


if __name__ == "__main__":
   main()
