import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

from utils import reverse_y_axis, detrend_time_series, estimate_trend, plot_predictions

def standardize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def train_ols(X, y):
    model = sm.OLS(y,X)
    results_constant = model.fit()
    return results_constant

def test_stationarity(series):
    adf_test = adfuller(series, regression="ct")
    return {'ADF Statistic': adf_test[0], 'p-value': adf_test[1], 'Stationary': adf_test[1] < 0.05}

def dtw_loss(true_signal, predicted_signal):
    dtw_distance, _ = fastdtw(np.expand_dims(true_signal, 1), np.expand_dims(predicted_signal, 1), dist=euclidean)
    return dtw_distance

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
        lagged_data[f'detrend_mnfm{lag}'] = df['detrend_MNF'].shift(lag, fill_value=df['detrend_MNF'][0])
        lagged_data[f'hrm{lag}'] = df['HR_mean'].shift(lag, fill_value=df['HR_mean'][0])
    
    lagged_df = pd.DataFrame(lagged_data)
    return pd.concat([df, lagged_df], axis=1)

def evaluate(model, y_true, y_pred):
    true_fatigue_decline = np.polyfit(np.arange(len(y_true)), y_true, 1)[0]
    pred_fatigue_decline = np.polyfit(np.arange(len(y_pred)), y_pred, 1)[0]
    return {"fatigue_error": (pred_fatigue_decline-true_fatigue_decline)**2, 
            "mse": mean_squared_error(y_true, y_pred), 
            "rmse": sqrt(mean_squared_error(y_true, y_pred)),
            "mape": mean_absolute_percentage_error(y_true, y_pred), 
            "mae": mean_absolute_error(y_true, y_pred), 
            "DTW-Loss": dtw_loss(y_true, y_pred), 
            "AIC": model.aic if hasattr(model, 'aic') else 'N/A',
            "BIC": model.bic if hasattr(model, 'bic') else 'N/A',
    }

def main():
    ids = [2,3,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22]
    path_to_trajectory_files = Path(__file__).parent / "data" / "temporary_files"
    trajectory_data = []
    path_to_demographics = Path(__file__).parent / "data" / "stai_questionnaire.csv"
    demographics = pd.read_csv(path_to_demographics, sep=";").loc[:, ['Participant ID', 'Height (cm)', 'Body mass (kg)', 'Age', 'Sex', 'Climbing experience (years)', 'Bouldering experience (years)', 'Approximate the number of hours that you spend lead climbing per week.', 'Approximate the number of hours that you spend top-rope climbing per week', 'Approximate the number of hours that you spend bouldering per week']]
    demographics["climbing hours"] = demographics[['Approximate the number of hours that you spend lead climbing per week.', 'Approximate the number of hours that you spend top-rope climbing per week']].sum(axis=1)
    demographics = demographics.rename(columns={'Approximate the number of hours that you spend bouldering per week': 'bouldering hours'})
    demographics['Sex'] = demographics['Sex'].apply(lambda x: 1 if x == 'Male' else 0)
    demographics['BMI'] = demographics['Body mass (kg)'] / (demographics['Height (cm)'] / 100) ** 2
    demographics = demographics.drop(['Body mass (kg)', 'Height (cm)'], axis=1)

    for trajectory_file_name in os.listdir(path_to_trajectory_files):
        participant = read_in_trajectories(path_to_trajectory_files / trajectory_file_name)
        participant['y'] = reverse_y_axis(participant['y'])
        participant['detrend_MNF'] = detrend_time_series(participant['MNF'])
        participant_demographics = demographics.loc[demographics["Participant ID"] == int(trajectory_file_name.split("_")[0]), :]
        df_repeated = pd.DataFrame(data=np.repeat(participant_demographics.values, len(participant), axis=0), columns=participant_demographics.columns)
        style = 1 if trajectory_file_name.split("_")[1] == "topropeclimbing.csv" else 0
        df_repeated["style"] = pd.Series([style] * len(participant))
        participant["t"] = participant.index
        trajectory_data.append(pd.concat([participant, df_repeated], axis=1))
        plt.plot(participant["MNF"]/100)
        plt.plot(participant["x"])
        plt.show()

    kf = LeaveOneOut()
    data = pd.concat(trajectory_data, axis = 0)
    data.to_csv("all_data.csv")
    participant_ids = list(range(len(trajectory_data)))
    cross_validation_results = []
    cross_validation_results_muxcat = []
    for lag_start, lag_end in zip((1,20), (2,21)):
        print(lag_start, lag_end)
        features_with = [f'acc_xm{lag}' for lag in range(lag_start, lag_end)] + [f'acc_ym{lag}' for lag in range(lag_start, lag_end)] + [f'xm{lag}' for lag in range(lag_start, lag_end)] + [f'ym{lag}' for lag in range(lag_start,lag_end)] +[f"muxcat_{i}" for i in range(1,17)] +[f'mnfm{lag}' for lag in range(lag_start,lag_end)]+["t"]
        features_without = [f'acc_xm{lag}' for lag in range(lag_start, lag_end)] + [f'acc_ym{lag}' for lag in range(lag_start, lag_end)] + [f'xm{lag}' for lag in range(lag_start, lag_end)] + [f'ym{lag}' for lag in range(lag_start,lag_end)] +[f'mnfm{lag}' for lag in range(lag_start,lag_end)] +["t"]
        for train_index, test_index in kf.split(participant_ids):
            train_data = [trajectory_data[i] for i in train_index]
            test_data = [trajectory_data[i] for i in test_index]
            trends = []
            for data in train_data:
                trend = estimate_trend(data["MNF"])
                trends.append(trend)
            average_trend = np.mean(trends, axis=0)
            X_train = pd.concat([create_lagged_features(df, lags=lag_end) for df in train_data], axis=0)
            expected_y = X_train[f"ym{lag_end}"].copy()
            expected_y[expected_y < 0] = 0
            muxcat_dummies = pd.get_dummies(expected_y.astype(int), drop_first=True, prefix="muxcat")*1
            if 'muxcat_18' in muxcat_dummies.columns:
                    muxcat_dummies = muxcat_dummies.drop('muxcat_18', axis=1)  
            X_train =  pd.concat([X_train, muxcat_dummies], axis=1)
            ols_model = train_ols(X_train[features_without], X_train["MNF"])
            ols_model_muxcat = train_ols(X_train[features_with], X_train["MNF"].tolist())   
            for df in test_data:
                df = create_lagged_features(df, lags=lag_end)

            all_results = []
            all_results_musxcat = []
            for df in test_data:
                y_test = df['MNF']
                df = create_lagged_features(df, lags=lag_end)
                expected_y = df[f"ym{lag_end}"].copy()
                expected_y[expected_y < 0] = 0
                muxcat_dummies = pd.get_dummies(expected_y.astype(int), drop_first=True, prefix="muxcat")*1
                if 'muxcat_18' in muxcat_dummies.columns:
                    muxcat_dummies = muxcat_dummies.drop('muxcat_18', axis=1)    
                df =  pd.concat([df, muxcat_dummies], axis=1)
                ols_predictions_muxcat = ols_model_muxcat.predict(df[features_with]) #+ (average_trend[1]+ average_trend[0] * np.arange(len(df[features_with])))
                ols_predictions = ols_model.predict(df[features_without]) #+  (average_trend[1] + average_trend[0] * np.arange(len(df[features_with])))
                #plot_predictions(y_test, ols_predictions_muxcat)
                #plot_predictions(y_test, ols_predictions)
                forecasting_results_muxcat = evaluate(ols_model_muxcat, y_test, ols_predictions_muxcat)
                forecasting_results = evaluate(ols_model, y_test, ols_predictions)
                all_results.append(forecasting_results)
                all_results_musxcat.append(forecasting_results_muxcat)

            cv_results = pd.DataFrame(all_results)
            cv_results_musxcat = pd.DataFrame(all_results_musxcat)
            cross_validation_results.append(cv_results)
            cross_validation_results_muxcat.append(cv_results_musxcat)

        term = "short"
        if lag_end == 21:
            term = "long"
        
        path_to_save_results = Path(__file__).parent / "data" / "results"
        evaluation_mean_df = pd.concat(cross_validation_results, axis=0).mean().round(3)
        evaluation_std_df = pd.concat(cross_validation_results, axis=0).std().round(3)
        print(evaluation_mean_df)
        #print(evaluation_std_df)

        evaluation_mean_df.to_csv(path_to_save_results / f"AR_forecast_results_mean_{term}.csv")
        evaluation_std_df.to_csv(path_to_save_results / f"AR_forecast_results_std_{term}.csv")
        print("WITH MUXCAT")
        evaluation_mean_df = pd.concat(cross_validation_results_muxcat, axis=0).mean().round(3)
        evaluation_std_df = pd.concat(cross_validation_results_muxcat, axis=0).std().round(3)
        print(evaluation_mean_df)
        #print(evaluation_std_df)

        evaluation_mean_df.to_csv(path_to_save_results / f"AR_forecast_results_mean_muxca_{term}.csv")
        evaluation_std_df.to_csv(path_to_save_results / f"AR_forecast_results_std_muxcat_{term}.csv")
if __name__ == "__main__":
    main()
