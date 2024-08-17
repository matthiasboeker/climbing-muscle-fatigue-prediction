# Predictive Modelling of Muscle Fatigue in Climbing

## Overview

This repository contains the code, data, and resources related to the study on predicting muscle fatigue during sport climbing. The study, "Predictive Modelling of Muscle Fatigue in Climbing" focuses on analyzing muscle fatigue in climbers and comparing various predictive models to forecast muscle fatigue 5 seconds into the future.

## Abstract

Sport climbing, a discipline demanding high levels of muscular strength, endurance, and cognitive planning, has gained considerable popularity in recent years. The importance of managing muscle fatigue during climbing, which can substantially impair performance and potentially lead to injury, has not yet been thoroughly investigated.

This study aimed to monitor and predict muscle fatigue during climbing using a multi-modal approach involving electromyography (EMG) and video tracking. We compared different linear Autoregressive (AR) methods and machine learning models, including Multi-Layer Perceptron (MLP) and Gradient Boosting (GB), based on their predictive performance.

While non-linear models outperformed linear methods, the simplicity of calculating linear models could enable real-time predictions on wearable devices, providing climbers with immediate feedback on muscle fatigue. This capability can significantly impact climbing research by providing practical tools for real-world applications, improving climbers' decision-making, and enhancing safety and performance.

## Repository Structure

- **data/**: Insert the data from according the OSF repository.
- **signal_processing/**: Includes the source code for data processing, feature extraction, and predictive modeling.
- **utils.py/**: Contains supporting functions for plotting, data transformations, and evaluation metrics.
- **manual_annotations.py/**: This script processes video data to annotate ground truth climbing trajectories by allowing manual annotation of the climber's center of gravity (COG) in each frame and saves the annotations as CSV files. 
- **ar_predictions.py/**: This script applies Autoregressive (AR) models to predict muscle fatigue from climbing trajectory and physiological data, performing cross-validation to evaluate model performance and save the results.
- **non_linear_predictions.py/**: This script implements and evaluates several machine learning models, including LSTM, ANN, CNN, Gradient Boosting, and Random Forest, for predicting muscle fatigue based on climbing trajectory and physiological data, using cross-validation to compare their performance.
- **README.md**: This file.

## Getting Started

### Prerequisites

The code uses conda to manage dependencies. The dependencies are given in yml file **environment.yml. 

You can install the necessary packages using pip:

```bash
conda env create -f environment.yml
