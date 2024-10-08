import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torch.optim as optim
from utils import dtw_loss, create_lagged_features, read_in_trajectories, plot_predictions, reverse_y_axis

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
    
class ANNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x    
    
    

class CNNModel(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, output_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size, stride=1, padding=kernel_size//2)
        self.fc = nn.Linear(num_filters * input_size, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_ols(X, y):
    model = sm.OLS(y, X)
    results = model.fit()
    return results

def train_rf(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def train_pytorch_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter if validation loss improves
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            patience_counter += 1  # Increment counter if no improvement
            if patience_counter >= patience:
                print('Early stopping triggered')
                model.load_state_dict(torch.load('best_model.pth'))  # Load the best model
                return model, best_val_loss

    model.load_state_dict(torch.load('best_model.pth'))  # Ensure the best model is loaded at the end
    return model, best_val_loss

def evaluate_forecast(y_true, y_pred):
    true_fatigue_decline = np.polyfit(np.arange(len(y_true)), y_true, 1)[0]
    pred_fatigue_decline = np.polyfit(np.arange(len(y_pred)), y_pred, 1)[0]
    return {"fatigue_error": (pred_fatigue_decline-true_fatigue_decline)**2, 
            "mse": mean_squared_error(y_true, y_pred), 
            "rmse": sqrt(mean_squared_error(y_true, y_pred)),
            "mape": mean_absolute_percentage_error(y_true, y_pred), 
            "mae": mean_absolute_error(y_true, y_pred), 
            #"DTW-Loss": dtw_loss(y_true, y_pred), 
    }
def plot_feature_importance(feature_importances, feature_names):
        # Sort feature importances in descending order and create labels accordingly
        indices = np.argsort(feature_importances)[::-1]
        names = [feature_names[i] for i in indices]
        values = feature_importances[indices]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(values)), values, align='center')
        plt.xticks(range(len(values)), names, rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
        plt.show()

def standardize_features(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, scaler

def standardize_y(y):
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
    return y_scaled, scaler
    

def main():
    path_to_trajectory_files = Path(__file__).parent / "data" / "temporary_files"
    trajectory_data = []
    path_to_demographics = Path(__file__).parent / "data" / "stai_questionnaire.csv"

    for trajectory_file_name in os.listdir(path_to_trajectory_files):
        participant = read_in_trajectories(path_to_trajectory_files / trajectory_file_name)
        participant['y'] = reverse_y_axis(participant['y'])
        trajectory_data.append(participant)

    df = pd.concat(trajectory_data, axis=0)
    kf = LeaveOneOut()
    participant_ids = list(range(len(trajectory_data)))
    cross_validation_results = []
    for lag_start, lag_end in zip((1,20), (12,32)):
        features = [f'acc_xm{lag}' for lag in range(lag_start, lag_end)] + [f'acc_ym{lag}' for lag in range(lag_start, lag_end)]+[f'xm{lag}' for lag in range(lag_start, lag_end)]+[f'ym{lag}' for lag in range(lag_start, lag_end)]+[f'mnfm{lag}' for lag in range(lag_start, lag_end)] 
        for train_index, test_index in kf.split(participant_ids):
            train_data = [trajectory_data[i] for i in train_index]
            test_data = [trajectory_data[i] for i in test_index]

            X_train = []

            for df in train_data:
                X_train.append(create_lagged_features(df, lags=lag_end))
            X_train_ = pd.concat(X_train, axis=0)
            print(X_train_.shape)
            X_train , X_scaler = standardize_features(X_train_[features])
            y_train, y_scaler = standardize_y(X_train_['MNF'])
            X_train_pytorch =X_train[features].values
            y_train_pytorch = y_train
            X_train_pytorch = torch.tensor(X_train_pytorch, dtype=torch.float32)
            y_train_pytorch = torch.tensor(y_train_pytorch, dtype=torch.float32)#.unsqueeze(1)
            train_dataset = torch.utils.data.TensorDataset(X_train_pytorch, y_train_pytorch)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

            X_val_list = []
            for df in test_data:
                X_val_list.append(create_lagged_features(df, lags=lag_end))
            X_val_ = pd.concat(X_val_list, axis=0)
            print(X_val_.shape)
            columns = X_val_[features].columns
            X_val =  pd.DataFrame(X_scaler.transform(X_val_[features]), columns=columns)
            y_val =  y_scaler.transform(X_val_["MNF"].values.reshape(-1,1))
            X_val_pytorch = X_val[features].values
            y_val_pytorch = y_val
            X_val_pytorch = torch.tensor(X_val_pytorch, dtype=torch.float32)
            y_val_pytorch = torch.tensor(y_val_pytorch, dtype=torch.float32)#.unsqueeze(1)
            val_dataset = torch.utils.data.TensorDataset(X_val_pytorch, y_val_pytorch)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Train LSTM Model
            lstm_model = LSTMModel(input_size=X_train_pytorch.shape[1], hidden_size=50, num_layers=2, output_size=1)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
            #print("Train LSTM")
            lstm_val_loss = train_pytorch_model(lstm_model, criterion, optimizer, train_loader, val_loader, num_epochs=1)

            # Train ANN Model
            ann_model = ANNModel(input_size=X_train_pytorch.shape[1], hidden_size=50, output_size=1)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(ann_model.parameters(), lr=0.001)
            #print("Train ANN")
            ann_val_loss = train_pytorch_model(ann_model, criterion, optimizer, train_loader, val_loader, num_epochs=1)

            #Train Random Forest
            # Initialize models
            gbm = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
            rf = RandomForestRegressor(n_estimators=10, max_depth=3)
            #print("TRAIN RF")
            gbm = train_rf(gbm, X_train[features].values, y_train.ravel())
            #plot_feature_importance(gbm.feature_importances_, features)
            rf = train_rf(rf, X_train[features].values, y_train.ravel())

            all_results = []
            for df in test_data:
                df = create_lagged_features(df, lags=lag_end)
                X_test =  pd.DataFrame(X_scaler.transform(df[features]), columns=features)
                X_test = X_test.reset_index(drop=True)
                y_test = df["MNF"].values
                # PyTorch model predictions
                X_test_pytorch = torch.tensor(X_test.values, dtype=torch.float32)
                lstm_predictions = y_scaler.inverse_transform(lstm_model(X_test_pytorch).detach().numpy().flatten().reshape(-1, 1))
                ann_predictions = y_scaler.inverse_transform(ann_model(X_test_pytorch).detach().numpy().flatten().reshape(-1, 1))

                gbm_predictions = y_scaler.inverse_transform(gbm.predict(X_test.values).reshape(-1, 1))
                rf_predictions = y_scaler.inverse_transform(rf.predict(X_test.values).reshape(-1, 1))

                lstm_forecasting_results = evaluate_forecast(y_test, lstm_predictions)
                ann_forecasting_results = evaluate_forecast(y_test, ann_predictions)
                gbm_forecasting_results = evaluate_forecast(y_test, gbm_predictions)
                rf_forecasting_results = evaluate_forecast(y_test, rf_predictions)

                all_results.append(('LSTM', lstm_forecasting_results))
                all_results.append(('ANN', ann_forecasting_results))
                all_results.append(('RF', rf_forecasting_results))
                all_results.append(('GBM', gbm_forecasting_results))

            results_df = pd.DataFrame([result for model_name, result in all_results], index=[model_name for model_name, result in all_results])
            cross_validation_results.append(results_df)
            #print("PLOT ANN")
            #plot_predictions(y_test, ann_predictions)
            #print("PLOT LSTM")
            #plot_predictions(y_test, lstm_predictions)
            #print("PLOT GB")
            #plot_predictions(y_test, gbm_predictions)
            #print("PLOT RF")
            #plot_predictions(y_test, rf_predictions)
        evaluation_mean_df = pd.concat(cross_validation_results).groupby(level=0).mean().round(3)
        evaluation_std_df = pd.concat(cross_validation_results).groupby(level=0).std().round(3)
        
        print(evaluation_mean_df)
        print(evaluation_std_df)
        path_to_save_results = Path(__file__).parent / "data" / "results"
        term = "short"
        if lag_end == 32:
            term = "long"
        evaluation_mean_df.to_csv(path_to_save_results / f"NN_forecast_results_mean_{term}.csv")
        evaluation_std_df.to_csv(path_to_save_results / f"NN_forecast_results_std_{term}.csv")

if __name__ == "__main__":
    main()
