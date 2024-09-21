# Predictive Maintenance Report: LSTM vs GRU on CMAPSS Dataset

## Objective
The objective of this study was to implement and compare the performance of Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) models on the FD001 dataset from the NASA CMAPSS Turbofan Engine dataset for predictive maintenance purposes. Additionally, we extended our analysis to the FD002 dataset to assess model performance under increased complexity.

## Part 1: Data Acquisition and Preprocessing

### Data Collection
We used the FD001 and FD002 datasets from the NASA CMAPSS Turbofan Engine dataset. These datasets contain sensor readings from aircraft engines, with FD001 representing a single operating condition and FD002 involving multiple operating conditions.

### Data Preprocessing
Our preprocessing pipeline included the following steps:

1. Loading the data using pandas, with columns for unit, cycle, operational settings, and sensor readings.
2. Removing constant columns (sensor_22 and sensor_23) as they don't provide useful information.
3. Normalizing the data using MinMaxScaler to ensure all features are on the same scale.
4. Calculating the Remaining Useful Life (RUL) for each engine by subtracting the current cycle from the maximum cycle for that unit.
5. Creating sequences of data points to capture temporal dependencies, with a sequence length of 180 determined through experimentation.

## Part 2: Model Building

### Model Development
We implemented both LSTM and GRU models using PyTorch. The models were structured as follows:

1. LSTM Model:
   - 3 LSTM layers with 128 hidden units each
   - Dropout of 0.3 between LSTM layers
   - Two fully connected layers (128 -> 64 -> 1)

2. GRU Model:
   - 3 GRU layers with 128 hidden units each
   - Dropout of 0.3 between GRU layers
   - Two fully connected layers (128 -> 64 -> 1)

### Feature Engineering
We incorporated the following feature engineering techniques:

1. Sequence creation: We used a sliding window approach to create sequences of 180 time steps, capturing temporal patterns in the data.
2. Normalization: All features were normalized using MinMaxScaler to ensure they were on the same scale.
3. RUL calculation: We engineered the RUL feature by calculating the remaining cycles for each engine.

## Part 3: Model Evaluation and Comparison

### Model Performance Evaluation
We evaluated both models using the following metrics:

1. Mean Squared Error (MSE)
2. Mean Absolute Error (MAE)
3. Accuracy
4. Precision
5. Recall

The results for FD001 dataset were:

LSTM:
- MSE: 3063.0784
- MAE: 43.186802
- Accuracy: 0.7740786621244967
- Precision: 0.7290836653386454
- Recall: 0.06033630069238378

GRU:
- MSE: 2468.8804
- MAE: 39.066135
- Accuracy: 0.7824403840198204
- Precision: 0.5916187345932621
- Recall: 0.23738872403560832

### Visualization
We created several visualizations to compare the performance of LSTM and GRU models:

1. Training history plots showing the training and validation loss for both models.
2. Actual vs. Predicted RUL plots for the first 100 data points.
3. Bar charts comparing MSE and MAE for both models.

These visualizations helped to illustrate the differences in performance between LSTM and GRU models.

## Part 4: Conclusion and Future Directions

### Summary
Based on our analysis of the FD001 dataset:

1. The GRU model outperformed the LSTM model in terms of MSE and MAE.
2. Both models achieved similar accuracy, with the GRU model slightly ahead.
3. The LSTM model showed higher precision, while the GRU model demonstrated better recall.

### Discussion
Advantages and disadvantages of LSTM and GRU in the context of predictive maintenance:

LSTM:
- Advantages: Better at capturing long-term dependencies, which can be crucial in predicting RUL.
- Disadvantages: More complex architecture, potentially leading to longer training times and increased risk of overfitting.

GRU:
- Advantages: Simpler architecture, faster training, and good performance on shorter sequences.
- Disadvantages: May struggle with very long-term dependencies compared to LSTM.

In our case, the GRU model's simpler architecture seemed to be beneficial, resulting in better overall performance on the FD001 dataset.

### Future Directions
1. Experiment with bidirectional LSTM/GRU models to capture both past and future context.
2. Implement attention mechanisms to help the models focus on the most relevant parts of the input sequences.
3. Explore ensemble methods, combining LSTM and GRU predictions for potentially improved performance.
4. Investigate the use of more advanced preprocessing techniques, such as wavelet transforms or Fourier analysis, to extract more informative features from the raw sensor data.

## Bonus Task: Comparison on FD002 Dataset

We extended our analysis to the FD002 dataset, which involves multiple operating conditions. The results were as follows:

LSTM on FD002:
- MSE: 3718.8162
- MAE: 48.8815
- Accuracy: 0.7658158587441957
- Precision: 0.8936170212765957
- Recall: 0.010510510510510511

GRU on FD002:
- MSE: 3708.0205
- MAE: 47.901756
- Accuracy: 0.7917245866729763
- Precision: 0.8637059724349158
- Recall: 0.14114114114114115

Comparing FD001 and FD002 results:

1. Complexity: FD002 involves six operating conditions compared to FD001's single condition, making it a more complex dataset.
2. Performance: Both LSTM and GRU models showed higher error rates on FD002, indicating difficulty in handling the increased complexity.
3. Model Comparison: The performance gap between LSTM and GRU is smaller on FD002, suggesting both models struggle similarly with the increased complexity.
4. Generalization: The models' ability to generalize across different operating conditions in FD002 is crucial for real-world applications.

Future Work:
Consider incorporating operating condition information more explicitly in the model architecture to better handle the complexity of FD002. This could involve:
1. Adding condition-specific embeddings or encodings.
2. Implementing a hierarchical model structure that first predicts the operating condition and then the RUL.
3. Using multi-task learning to predict both the operating condition and RUL simultaneously.

In conclusion, while both LSTM and GRU models showed promise in predicting RUL for aircraft engines, there is still room for improvement, especially when dealing with more complex scenarios involving multiple operating conditions. Future work should focus on developing more sophisticated architectures and preprocessing techniques to better capture the intricacies of real-world predictive maintenance challenges.
