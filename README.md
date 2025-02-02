# Air Quality Forecasting Using LSTM

## Project Overview

This project aims to forecast air quality, specifically PM2.5 concentration levels, in Beijing using historical air quality and weather data. The dataset contains hourly measurements, and the goal is to predict future PM2.5 concentrations using a time series forecasting approach with Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models.

The project includes data preprocessing, feature engineering, model design, and experimentation with various hyperparameters and architectures. The models are evaluated using RMSE (Root Mean Squared Error), and predictions are submitted to the Kaggle competition for further evaluation.

## Data

The dataset used in this project is divided into the following files:

- `train.csv`: Training data with historical air quality and weather measurements.
- `test.csv`: Test data used for making predictions.
- `sample_submission.csv`: A sample submission file that includes placeholders for the final predictions.

## Kaggle Competition

This project is part of the Kaggle competition titled "Beijing PM2.5 Forecasting". The competition can be accessed here: https://www.kaggle.com/competitions/alu-jan25-air-quality-forecasting/overview.

Leaderboard: The model ranks in the top 20% based on the submission file `submission.csv`.

## Requirements

Before running the code, ensure the following Python libraries are installed:
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- scikit-learn

You can install the required libraries by running:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

## Experimentation and Results

Throughout the project, multiple experiments were conducted to optimize the LSTM model for forecasting PM2.5 concentrations. The following table summarizes the experiments, including different hyperparameters, model configurations, and the resulting RMSE values.

| Experiment # | Learning Rate | LSTM Layers | Units per Layer | Batch Size | Dropout | Sequence Length | Activation | Optimizer | Final Training Loss (MSE) | RMSE    |
|--------------|---------------|-------------|-----------------|------------|---------|-----------------|------------|-----------|---------------------------|---------|
| 1 (Default)  | N/A           | 2           | 128, 64         | 32         | None    | 24              | ReLU       | Adam      | 1978.88                   | 48.48   |
| 2            | 0.01          | 2           | 128, 64         | 32         | None    | 24              | ReLU       | Adam      | 2759.32                   | 52.53   |
| 3            | 0.001         | 3           | 128, 64, 32     | 32         | None    | 24              | ReLU       | Adam      | 3993.91                   | 63.20   |
| 4            | 0.0001        | 2           | 64, 32          | 64         | 0.2     | 24              | Tanh       | Adam      | 12922.91                  | 113.68  |
| 5            | 0.001         | 3           | 64, 32, 16      | 32         | 0.2     | 24              | ReLU       | Adam      | 3514.13                   | 59.28   |
| 6            | 0.001         | 1           | 64              | 32         | None    | 24              | ReLU       | Adam      | 3168.43                   | 56.29   |
| 7            | 0.001         | 2           | 128, 64         | 64         | None    | 24              | Tanh       | Adam      | 2312.82                   | 48.09   |
| 8            | 0.001         | 2           | 64, 32          | 32         | 0.2     | 24              | Tanh       | RMSprop   | 2483.25                   | 49.83   |
| 9            | 0.001         | 2           | 128, 64         | 32         | 0.3     | 24              | ReLU       | RMSprop   | 2901.45                   | 53.85   |
| 10           | 0.001         | 2           | 64, 32          | 32         | None    | 12              | ReLU       | Adam      | 2901.84                   | 53.85   |
| 11           | 0.001         | 2           | 128, 64         | 32         | None    | 48              | ReLU       | RMSprop   | 3000.01                   | 54.77   |
| 12           | 0.001         | 2           | 64, 32          | 32         | None    | 24              | Sigmoid    | Adam      | 3200.00                   | 56.57   |
| 13           | 0.001         | 2           | 128, 64         | 32         | None    | 24              | Tanh       | SGD (Early Stopping) | 3600.62 | 60.00   |
| 14           | 0.001         | 2           | 64, 32          | 32         | None    | 24              | ReLU       | Adam (Batch Norm)   | 2854.65 | 52.91   |
| 15           | 0.001         | 2           | 128, 64         | 32         | None    | 24              | ReLU       | Adam (Batch Norm + Early Stopping) | 2750.57 | 52.43   |

## Results and Discussion

The best-performing model configuration, as determined by the experimentation process, resulted in an RMSE of 48.48. The model was optimized using a combination of techniques, including:

- Different LSTM layer configurations.
- Hyperparameter tuning for learning rate, batch size, and sequence length.
- Use of Dropout, Batch Normalization, and Early Stopping to prevent overfitting.
The final submission was ranked in the top 20% on the Kaggle leaderboard, demonstrating the effectiveness of the chosen model and configuration.