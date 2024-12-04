import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
file_path = 'preprocessed_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Define high exposure threshold (example: 75th percentile for each gas)
thresholds = {
    'NO2': data['NO2 Mean'].quantile(0.75),
    'O3': data['O3 Mean'].quantile(0.75),
    'SO2': data['SO2 Mean'].quantile(0.75),
    'CO': data['CO Mean'].quantile(0.75)
}

# Create binary treatment variables
data['NO2_Treated'] = (data['NO2 Mean'] > thresholds['NO2']).astype(int)
data['O3_Treated'] = (data['O3 Mean'] > thresholds['O3']).astype(int)
data['SO2_Treated'] = (data['SO2 Mean'] > thresholds['SO2']).astype(int)
data['CO_Treated'] = (data['CO Mean'] > thresholds['CO']).astype(int)

# Define confounders (example: demographic variables)
confounders = ['Age', 'Income', 'Smoking', 'PhysicalActivity']


# Function to calculate propensity scores and validate them
def calculate_propensity_scores(data, confounders, treatment_column, gas_name):
    # Define features (confounders) and target (treatment variable)
    X = data[confounders]
    y = data[treatment_column]

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Calculate propensity scores
    data[f'{gas_name}_Propensity'] = model.predict_proba(X)[:, 1]

    # Evaluate model with AUC score
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"AUC Score for {gas_name} Propensity Model: {auc_score:.2f}")

    # Visualize propensity score overlap
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data[data[treatment_column] == 1][f'{gas_name}_Propensity'], label='Treated', color='blue')
    sns.kdeplot(data[data[treatment_column] == 0][f'{gas_name}_Propensity'], label='Untreated', color='orange')
    plt.title(f'Propensity Score Overlap for {gas_name}')
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y, data[f'{gas_name}_Propensity'], n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Perfect Calibration')
    plt.title(f'Calibration Plot for {gas_name} Propensity Scores')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid(True)
    plt.show()


# Run propensity score calculation for each gas
for gas, treatment_column in zip(['NO2', 'O3', 'SO2', 'CO'],
                                 ['NO2_Treated', 'O3_Treated', 'SO2_Treated', 'CO_Treated']):
    calculate_propensity_scores(data, confounders, treatment_column, gas)

# Save the data with propensity scores
data.to_csv('data_with_propensity_scores.csv', index=False)
print("Data with propensity scores saved as 'data_with_propensity_scores.csv'")
