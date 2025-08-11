# Battery State of Health (SoH) Estimation using Deep Learning

## 📖 Overview

This repository contains the code and resources for a project focused on estimating the State of Health (SoH) of lithium-ion batteries. The project utilizes the NASA Battery Aging Dataset and implements several deep learning models to predict battery degradation. The core of this work lies in advanced feature engineering and a comparative analysis of different neural network architectures to achieve robust and accurate SoH estimation.

The primary models explored in this project are:
- **Long Short-Term Memory (LSTM)**
- **Bidirectional LSTM (BiLSTM)**
- **Temporal Convolutional Network (TCN)**

## 📊 Dataset

This project uses the **NASA Battery Dataset**, which is publicly available on Kaggle. This dataset contains information from a battery prognostics experiment, including charge and discharge cycles under different operational conditions.

- **Dataset Link**: [NASA Battery Dataset on Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)

## ✨ Features

A key aspect of this project is feature engineering to extract meaningful information from the raw battery data. The features used for training the models include:

- **Target Variable**:
    - `calculated_capacity`

- **Cycle-based Features**:
    - `voltage_start`, `voltage_end`, `voltage_mean`, `voltage_min`, `voltage_drop`
    - `current_mean`, `current_std`
    - `temperature_mean`, `temperature_max`, `temperature_range`
    - `duration_seconds`, `duration_hours`
    - `energy_wh` (Energy in Watt-hours)
    - `ambient_temperature`
    - `dq_dv_max`
    - `voltage_variance`
    - `coulombic_efficiency`

- **Impedance Features**:
    - `Re_mapped` (Ohmic resistance)
    - `Rct_mapped` (Charge-transfer resistance)
    - `impedance_time_diff`


- **Time-Series Features**:
    - Interpolated voltage, current, and temperature profiles over a normalized time axis.

## 🧠 Models

Three deep learning models were implemented and compared for their effectiveness in SoH estimation:

1.  **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) well-suited for time-series data, capable of learning long-term dependencies.
2.  **BiLSTM (Bidirectional LSTM)**: An extension of LSTM that processes the input sequence in both forward and backward directions, allowing it to capture context from both past and future states.
3.  **TCN (Temporal Convolutional Network)**: A convolutional neural network (CNN) architecture designed for sequence modeling, often outperforming RNNs in terms of computational efficiency and performance on long sequences.

## 🚀 Setup and Installation

To run this project, you'll need to set up a Python environment and install the required dependencies.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/1bytess/escl-soh-prediction.git
    cd escl-soh-prediction
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: A `requirements.txt` file should be created containing all the necessary libraries such as `pandas`, `numpy`, `tensorflow`, `scikit-learn`, `matplotlib`, `seaborn`, and `tqdm`.*

## 💻 Usage

The main logic of the project is contained within the `SoH_Prediction.ipynb` Jupyter Notebook. To run the notebook and reproduce the results, follow these steps:

1.  **Download the dataset** from the [Kaggle link](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) and place the `cleaned_dataset` folder in the root directory of the project.

2.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

3.  Open `SoH_Prediction.ipynb` and run the cells sequentially to perform data loading, preprocessing, feature engineering, model training, and evaluation.

## 📈 Results

The models were evaluated based on Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²) metrics. The performance of the models was compared to determine the most effective architecture for this task.

| Model  | MAE (Ah) | RMSE (Ah) | R²   | MAPE (%) |
| :----- | :------- | :-------- | :--- | :------- |
| **BiLSTM** | 0.0855   | 0.1314    | 0.8628 | 8.34     |
| **TCN** | 0.0728   | 0.1222    | 0.8813 | 7.05     |
| **LSTM** | 0.0681   | 0.1068    | 0.9094 | 6.25     |

The **Temporal Convolutional Network (TCN)** demonstrated the best performance across all metrics, suggesting its strong capability in capturing the temporal dynamics of battery degradation.


## 📁 File Structure
```
.
├── SoH_Prediction.ipynb      # Main Jupyter Notebook with all the code
├── cleaned_dataset/          # Directory for the NASA battery data
│   ├── data/
│   └── metadata.csv
├── export/                   # Directory for saved models, plots, and processed data
│   ├── models/
│   ├── plots/
│   └── processed_data/
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```