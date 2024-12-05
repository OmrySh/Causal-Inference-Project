import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# Step 1: Define Pre-Treatment and Post-Treatment Periods
def define_treatment_periods(data, treatment_year, pre_years, post_years):
    """
    Adds a column to classify individuals as pre-treatment or post-treatment.

    Parameters:
    - data: DataFrame
    - treatment_year: The year treatment occurred (e.g., high SO2 exposure)
    - pre_years: Number of years before treatment to classify as pre-treatment
    - post_years: Number of years after treatment to classify as post-treatment

    Returns:
    - data: Updated DataFrame with 'Period' column
    """
    data['Period'] = np.where(data['Year'] < treatment_year, 'Pre-Treatment',
                              np.where(data['Year'] > treatment_year, 'Post-Treatment', 'Treatment-Year'))
    return data


# Step 2: Clean and Prepare Data
def clean_and_prepare_data(data, confounders):
    """Standardize confounder variables and drop rows with missing data."""
    data = data.dropna(subset=confounders + ['PHYSHLTH'])
    scaler = StandardScaler()
    data.loc[:, confounders] = scaler.fit_transform(data[confounders])
    return data


# Step 3: Perform Matching
def match_pre_post(data, confounders, k=1):
    """
    Matches pre-treatment individuals to post-treatment individuals using KNN.

    Parameters:
    - data: DataFrame
    - confounders: List of confounders
    - k: Number of nearest neighbors (default 1 for one-to-one matching)

    Returns:
    - matched_data: DataFrame with matched individuals
    """
    # Split data into pre-treatment and post-treatment
    pre_data = data[data['Period'] == 'Pre-Treatment'].copy()
    post_data = data[data['Period'] == 'Post-Treatment'].copy()

    # Extract features for matching
    X_pre = pre_data[confounders].values
    X_post = post_data[confounders].values

    # Fit KNN model for matching
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_post)
    distances, indices = knn.kneighbors(X_pre)

    # Match individuals
    matched_pre = pre_data.iloc[np.arange(len(indices))]
    matched_post = post_data.iloc[indices[:, 0]]  # Only select the closest match (k=1)

    # Add a group column
    matched_pre['Matched_Group'] = 'Pre-Treatment'
    matched_post['Matched_Group'] = 'Post-Treatment'

    # Combine matched data
    matched_data = pd.concat([matched_pre, matched_post], ignore_index=True)
    return matched_data


# Step 4: Calculate Change in PHYSHLTH
def calculate_change_in_physhlth(data):
    """
    Calculate the change in PHYSHLTH for matched individuals/groups.

    Parameters:
    - data: DataFrame with matched individuals

    Returns:
    - change: Average change in PHYSHLTH
    """
    # Split data into pre and post treatment
    pre_physhlth = data[data['Matched_Group'] == 'Pre-Treatment']['PHYSHLTH'].values
    post_physhlth = data[data['Matched_Group'] == 'Post-Treatment']['PHYSHLTH'].values

    # Ensure the same number of individuals
    if len(pre_physhlth) != len(post_physhlth):
        raise ValueError("Mismatch between pre-treatment and post-treatment group sizes.")

    # Calculate average change
    change = np.mean(post_physhlth - pre_physhlth)
    return change


from scipy.stats import ttest_rel


def perform_paired_ttest(data):
    """
    Perform a paired t-test on pre-treatment and post-treatment PHYSHLTH.

    Parameters:
    - data: DataFrame with matched individuals/groups

    Returns:
    - t_stat: t-statistic
    - p_value: p-value for the test
    """
    # Extract pre-treatment and post-treatment PHYSHLTH values
    pre_physhlth = data[data['Matched_Group'] == 'Pre-Treatment']['PHYSHLTH'].values
    post_physhlth = data[data['Matched_Group'] == 'Post-Treatment']['PHYSHLTH'].values

    # Perform paired t-test
    t_stat, p_value = ttest_rel(post_physhlth, pre_physhlth)
    return t_stat, p_value


# Main Execution
if __name__ == "__main__":
    # Load Dataset
    file_path = 'data/questionnaires_data.csv'  # Replace with your dataset path
    data = pd.read_csv(file_path)

    # Define Treatment Periods (e.g., Treatment in 2009)
    data = define_treatment_periods(data, treatment_year=2009, pre_years=1, post_years=1)

    # Confounders
    confounders = [
        'GENHLTH', 'QSTVER', 'POORHLTH', 'QSTLANG', 'SMOKDAY2',
        'WEIGHT2', 'MARITAL', 'EXERANY2', 'SMOKE100', 'MENTHLTH',
        'EDUCA', 'HEIGHT3', 'PREGNANT', 'CHILDREN', 'Month'
    ]

    # Clean and Prepare Data
    data = clean_and_prepare_data(data, confounders)

    # Perform Matching
    matched_data = match_pre_post(data, confounders, k=1)

    # Calculate Change in PHYSHLTH
    change = calculate_change_in_physhlth(matched_data)
    print(f"Average Change in PHYSHLTH: {change:.2f} days")

    # Perform the paired t-test
    t_stat, p_value = perform_paired_ttest(matched_data)
    print(f"Paired t-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.4f}")

    # Interpretation
    if p_value < 0.05:
        print("The change in PHYSHLTH is statistically significant.")
    else:
        print("The change in PHYSHLTH is not statistically significant.")
