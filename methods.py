import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.utils import resample


# Clean and Prepare Data
def clean_and_prepare_data(data, confounders):
    """Standardize confounder variables and drop rows with missing data."""
    data = data.dropna(subset=confounders + ['PHYSHLTH'])
    scaler = StandardScaler()
    data.loc[:, confounders] = scaler.fit_transform(data[confounders])
    return data


# Perform Matching
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


# perform t-test
def perform_paired_ttest(pre_physhlth, post_physhlth):
    """
    Perform a paired t-test between pre-treatment and post-treatment PHYSHLTH.
    :param pre_physhlth:
    :param post_physhlth:
    :return:
    """

    # Perform paired t-test
    t_stat, p_value = ttest_rel(post_physhlth, pre_physhlth)

    print("Paired t-test results:")
    print(f"t-statistic: {t_stat}")
    # Interpretation
    if p_value < 0.05:
        print("The change in PHYSHLTH is statistically significant.")
    else:
        print("The change in PHYSHLTH is not statistically significant.")

    return t_stat, p_value


def calculate_physhlth_averages(data):
    """
    Calculates average PHYSHLTH for each period and year.

    Parameters:
    - data: DataFrame with treatment periods defined

    Returns:
    - DataFrame with averages grouped by period and year
    """
    return data.groupby(['Period', 'Month', 'Year'])['PHYSHLTH'].mean().reset_index()


def plot_physhlth_trends(data):
    """
    Plots PHYSHLTH averages over time for each treatment period, limited to years 2008-2010 and monthly granularity.

    Parameters:
    - data: DataFrame with treatment periods defined
    - start_year: Start year for filtering (default: 2008)
    - end_year: End year for filtering (default: 2010)
    """
    # Create the plot
    plt.figure(figsize=(10, 6))
    for period, group_data in data.groupby('Period'):
        # Combine year and month for better granularity on the x-axis
        time_axis = group_data['Year'] + (group_data['Month'] - 1) / 12
        plt.plot(
            time_axis, group_data['PHYSHLTH'], marker='o', label=period
        )
    plt.title('Average PHYSHLTH by Treatment Period (Monthly Granularity)')
    plt.xlabel('Year')
    plt.ylabel('Average PHYSHLTH (Unhealthy Days)')
    plt.legend(title="Period")
    plt.grid()
    plt.show()


def bootstrap_ci(data1, data2, n_iterations=1000, alpha=0.05):
    """
    Performs bootstrapping to calculate confidence intervals for mean differences.

    Parameters:
    - data1: Array of pre-treatment PHYSHLTH
    - data2: Array of post-treatment PHYSHLTH
    - n_iterations: Number of bootstrap iterations
    - alpha: Significance level

    Returns:
    - mean_diff: Mean difference
    - conf_int: Confidence interval as a tuple (lower, upper)
    """
    differences = []
    for _ in range(n_iterations):
        resampled_data1 = resample(data1)
        resampled_data2 = resample(data2)
        differences.append(np.mean(resampled_data2 - resampled_data1))
    lower = np.percentile(differences, 100 * (alpha / 2))
    upper = np.percentile(differences, 100 * (1 - alpha / 2))
    return np.mean(differences), (lower, upper)


def summarize_results(t_stat, p_value, mean_diff, conf_int):
    """
    Prints and summarizes results from the paired t-test and bootstrapping.

    Parameters:
    - t_stat: t-statistic
    - p_value: p-value
    - mean_diff: Mean difference
    - conf_int: Confidence interval
    """
    print("Paired t-test Results:")
    print(f"t-statistic: {t_stat:.2f}")
    print(f"p-value: {p_value:.4f}")
    print("\nMean Difference and Confidence Interval:")
    print(f"Mean Difference: {mean_diff:.2f}")
    print(f"95% Confidence Interval: ({conf_int[0]:.2f}, {conf_int[1]:.2f})")


def filter_and_label_treatment(data, state, event_year):
    """
    Filters the data to include only rows from the specified state and
    the year before, the event year, and the year after. Labels rows as
    pre-treatment, treatment, and post-treatment.

    Parameters:
    - data: DataFrame containing the dataset
    - state: The country/state to filter by
    - event_year: The year when the treatment/event occurred

    Returns:
    - filtered_data: Filtered and labeled DataFrame
    """
    # Filter by country and relevant years
    filtered_data = data[(data['State'] == state) &
                         (data['Year'] >= event_year - 1) &
                         (data['Year'] <= event_year + 1)]

    # Label rows based on the treatment period
    filtered_data['Period'] = np.where(
        filtered_data['Year'] < event_year, 'Pre-Treatment',
        np.where(filtered_data['Year'] > event_year, 'Post-Treatment', 'Treatment-Year')
    )

    return filtered_data


def pipline(data, state, event_year):
    filtered_data = filter_and_label_treatment(data, state=state, event_year=event_year)

    # Confounders
    confounders = ['GENHLTH', 'QSTVER', 'POORHLTH', 'QSTLANG', 'SMOKDAY2',
                   'WEIGHT2', 'MARITAL', 'EXERANY2', 'SMOKE100', 'MENTHLTH',
                   'EDUCA', 'HEIGHT3', 'PREGNANT', 'CHILDREN']

    # Clean and Prepare Data
    prep_data = clean_and_prepare_data(data=filtered_data, confounders=confounders)

    # Perform Matching
    matched_data = match_pre_post(data=prep_data, confounders=confounders, k=1)

    # Calculate Change in PHYSHLTH
    change = calculate_change_in_physhlth(data=matched_data)
    print(f"Average Change in PHYSHLTH: {change:.2f} days")

    # Calculate averages for visualization
    avg_physhlth = calculate_physhlth_averages(data=matched_data)

    # Plot PHYSHLTH trends
    plot_physhlth_trends(data=avg_physhlth)

    # Perform paired t-test
    pre_physhlth = matched_data[matched_data['Period'] == 'Pre-Treatment']['PHYSHLTH'].values
    post_physhlth = matched_data[matched_data['Period'] == 'Post-Treatment']['PHYSHLTH'].values

    # Calculate confidence intervals using bootstrapping
    mean_diff, conf_int = bootstrap_ci(data1=pre_physhlth, data2=post_physhlth)
    t_stat, p_value = perform_paired_ttest(pre_physhlth=pre_physhlth, post_physhlth=post_physhlth)

    # Summarize results
    summarize_results(t_stat, p_value, mean_diff, conf_int)


# Main Execution
if __name__ == "__main__":
    # Load Dataset
    file_path = 'data/questionnaires_data.csv'  # Replace with your dataset path
    questionnaires_data = pd.read_csv(file_path)

    # Run event analysis
    pipline(data=questionnaires_data, state='Connecticut', event_year=2009)
