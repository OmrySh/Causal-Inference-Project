import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.utils import resample
import statsmodels.formula.api as smf


# Clean and Prepare Data
def clean_and_prepare_data(data, confounders):
    """Standardize confounder variables and drop rows with missing data."""
    data = data.dropna(subset=confounders + ['GENHLTH'])
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


# Step 4: Calculate Change in GENHLTH
def calculate_change_in_physhlth(data):
    """
    Calculate the change in GENHLTH for matched individuals/groups.

    Parameters:
    - data: DataFrame with matched individuals

    Returns:
    - change: Average change in GENHLTH
    """
    # Split data into pre and post treatment
    pre_physhlth = data[data['Matched_Group'] == 'Pre-Treatment']['GENHLTH'].values
    post_physhlth = data[data['Matched_Group'] == 'Post-Treatment']['GENHLTH'].values

    # Ensure the same number of individuals
    if len(pre_physhlth) != len(post_physhlth):
        raise ValueError("Mismatch between pre-treatment and post-treatment group sizes.")

    # Calculate average change
    change = np.mean(post_physhlth - pre_physhlth)
    return change


# perform t-test
def perform_paired_ttest(pre_physhlth, post_physhlth):
    """
    Perform a paired t-test between pre-treatment and post-treatment GENHLTH.
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
        print("The change in GENHLTH is statistically significant.")
    else:
        print("The change in GENHLTH is not statistically significant.")

    return t_stat, p_value


def calculate_physhlth_averages(data):
    """
    Calculates average GENHLTH for each period and year.

    Parameters:
    - data: DataFrame with treatment periods defined

    Returns:
    - DataFrame with averages grouped by period and year
    """
    return data.groupby(['Period', 'Month', 'Year'])['GENHLTH'].mean().reset_index()


import matplotlib.dates as mdates


def plot_physhlth_trends(data):
    """
    Plots GENHLTH averages over time for each treatment period.

    Parameters:
    - data: DataFrame with Period, Year, Month, and GENHLTH columns.
    """
    # Convert Year and Month into a datetime (using the first day of each month)
    data['Date'] = pd.to_datetime(dict(year=data['Year'], month=data['Month'], day=1))

    plt.figure(figsize=(10, 6))
    for period, group_data in data.groupby('Period'):
        # If this is the treatment period, add 0.05 to GENHLTH values
        if period == 'Treatment-Month':
            genhlth_values = group_data['GENHLTH'] + 0.05
        else:
            genhlth_values = group_data['GENHLTH']

        group_data = group_data.sort_values(['Year', 'Month'])
        # Plot using the new Date column
        plt.plot(group_data['Date'], genhlth_values, marker='o', label=period)

    plt.title('Average GENHLTH by Treatment Period (Monthly Granularity)')
    plt.xlabel('Date')
    plt.ylabel('Average GENHLTH (Unhealthy Days)')
    plt.legend(title="Period")
    plt.grid()

    # Format the x-axis to show years and months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # Optionally set a locator for better spacing (e.g., every 3 months)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    plt.show()


def bootstrap_ci(data1, data2, n_iterations=1000, alpha=0.05):
    """
    Performs bootstrapping to calculate confidence intervals for mean differences.

    Parameters:
    - data1: Array of pre-treatment GENHLTH
    - data2: Array of post-treatment GENHLTH
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


def filter_and_label_treatment(data, state, event_month, event_year):
    """
    Filters the data to include only rows from the specified state and
    the month before, the event month, and the month after. Labels rows as
    pre-treatment, treatment, and post-treatment with monthly granularity.

    Parameters:
    - data: DataFrame containing the dataset
    - state: The country/state to filter by
    - event_month: The month when the treatment/event occurred (1–12)
    - event_year: The year when the treatment/event occurred

    Returns:
    - filtered_data: Filtered and labeled DataFrame with a 'Period' column
    """
    # Filter by state and relevant years
    filtered_data = data[data['State'] == state].copy()

    # Create a datetime column for accurate comparisons
    filtered_data['Date'] = pd.to_datetime(
        filtered_data['Year'].astype(str) + '-' + filtered_data['Month'].astype(str) + '-01'
    )

    # Define treatment date and bounds for one year before and after
    event_date = pd.Timestamp(f"{event_year}-{event_month:02d}-01")
    start_date = event_date - pd.DateOffset(months=11)
    end_date = event_date + pd.DateOffset(months=11)

    # Filter data within the one-year range
    filtered_data = filtered_data[(filtered_data['Date'] >= start_date) & (filtered_data['Date'] <= end_date)]

    # Label rows based on the treatment period
    filtered_data['Period'] = np.where(
        filtered_data['Date'] < event_date, 'Pre-Treatment',
        np.where(filtered_data['Date'] == event_date, 'Treatment-Month', 'Post-Treatment')
    )

    return filtered_data


# Step 1: Filter and Prepare Data
def filter_and_prepare_data(data, treated_state, treatment_month, treatment_year, confounders):
    """
    Filters the dataset for treated and control groups, defines monthly treatment periods,
    and preprocesses confounders.

    Parameters:
    - data: DataFrame containing the dataset
    - treated_state: State exposed to high SO₂ levels (e.g., 'Connecticut')
    - treatment_month: Month when treatment began (1–12)
    - treatment_year: Year when treatment began
    - confounders: List of confounder variable names

    Returns:
    - prepared_data: Filtered and labeled DataFrame with preprocessed confounders
    """
    # Define control states as all states except the treated state
    control_states = data['State'].unique()
    control_states = [state for state in control_states if state != treated_state]

    # Filter data for treated and control states
    filtered_data = data[
        ((data['State'] == treated_state) | (data['State'].isin(control_states))) &
        ((data['Year'] >= treatment_year - 1) | (
                (data['Year'] == treatment_year) & (data['Month'] <= treatment_month))) &
        ((data['Year'] <= treatment_year + 1) | (
                (data['Year'] == treatment_year + 1) & (data['Month'] > treatment_month)))
        ].copy()

    # Create a datetime column for monthly comparison
    filtered_data['Date'] = pd.to_datetime(
        filtered_data['Year'].astype(str) + '-' + filtered_data['Month'].astype(str) + '-01')

    # Define treatment periods: pre-treatment or post-treatment
    treatment_date = pd.Timestamp(f"{treatment_year}-{treatment_month:02d}-01")
    filtered_data['Post_Treatment'] = np.where(filtered_data['Date'] >= treatment_date, 1, 0)

    # Define treatment indicator
    filtered_data['Treatment'] = np.where(filtered_data['State'] == treated_state, 1, 0)

    # Interaction term for DiD
    filtered_data['Treatment_x_Post_Treatment'] = filtered_data['Treatment'] * filtered_data['Post_Treatment']

    # Drop rows with missing values for confounders
    filtered_data = filtered_data.dropna(subset=confounders + ['GENHLTH']).copy()

    # Standardize confounders
    scaler = StandardScaler()
    filtered_data[confounders] = scaler.fit_transform(filtered_data[confounders])

    return filtered_data


# Step 2: Perform DiD Analysis
def run_did_analysis(data, confounders):
    """
    Runs Difference-in-Differences regression analysis with confounders.

    Parameters:
    - data: Filtered and prepared DataFrame
    - confounders: List of confounder variable names

    Returns:
    - model: Fitted regression model
    """
    # Create the regression formula
    formula = 'GENHLTH ~ Treatment + Post_Treatment + Treatment_x_Post_Treatment + ' + ' + '.join(confounders)

    # Fit the regression model
    model = smf.ols(formula=formula, data=data).fit()
    return model


def visualize_trends(data, event_date):
    """
    Visualizes trends in GENHLTH for treated and control groups over time (monthly),
    splitting the plot into one color before the treatment day and another color after.

    Parameters:
    - data: Filtered DataFrame prepared for DiD analysis
            Must contain columns: 'Treatment', 'Year', 'Month', 'GENHLTH'
    - event_date: A datetime object indicating the treatment date
    """
    # Calculate average GENHLTH by group and time
    trends = (
        data.groupby(['Treatment', 'Year', 'Month'])['GENHLTH']
            .mean()
            .reset_index()
    )
    trends['Date'] = pd.to_datetime(trends['Year'].astype(str) + '-' +
                                    trends['Month'].astype(str) + '-01')

    # Plot trends
    plt.figure(figsize=(12, 6))
    for treatment, group in trends.groupby('Treatment'):
        # Define labels
        label = 'Treated Group' if treatment == 1 else 'Control Group'

        # Split into pre and post treatment
        pre_data = group[group['Date'] < event_date]
        post_data = group[group['Date'] > event_date]

        # Plot the pre-treatment segment in one color (e.g., blue)
        if not pre_data.empty:
            plt.plot(pre_data['Date'], pre_data['GENHLTH'], marker='o',
                     color='blue', label=label if treatment == 1 else label)

        # Plot the post-treatment segment in another color (e.g., red)
        if not post_data.empty:
            # We don't need to repeat the label here to avoid duplicating legend entries
            plt.plot(post_data['Date'], post_data['GENHLTH'], marker='o', color='red')

    plt.title('Trends in GENHLTH for Treated and Control Groups (Monthly)')
    plt.xlabel('Date')
    plt.ylabel('Average GENHLTH (Unhealthy Days)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def drop_dot_from_features_names(df, confounders):
    for i in range(len(confounders)):
        if '.' in confounders[i]:
            confounders[i] = confounders[i].replace('.', '_')
    df.columns = df.columns.str.replace('.', '_')
    return df, confounders


def pipline(data, state, event_year, event_month):
    # Confounders
    confounders = [
        'PHYSHLTH', 'POORHLTH', 'SMOKDAY2', 'WEIGHT2', 'EXERANY2',
        'SMOKE100', 'MENTHLTH', 'EDUCA', 'HEIGHT3', 'PREGNANT',
        'CHILDREN', 'MARITAL_1.0', 'MARITAL_2.0', 'MARITAL_3.0',
        'MARITAL_4.0', 'MARITAL_5.0', 'MARITAL_6.0'
    ]

    data, confounders = drop_dot_from_features_names(data, confounders)
    filtered_data = filter_and_label_treatment(data, state=state, event_month=event_month, event_year=event_year)

    # Clean and Prepare Data
    prep_data = clean_and_prepare_data(data=filtered_data, confounders=confounders)

    # Perform Matching
    matched_data = match_pre_post(data=prep_data, confounders=confounders, k=1)

    print("--------------------------")
    print("Matching analysis results:")
    print("--------------------------")

    # Calculate Change in GENHLTH
    change = calculate_change_in_physhlth(data=matched_data)
    print(f"Average Change in GENHLTH: {change:.2f} days")

    # Calculate averages for visualization
    avg_physhlth = calculate_physhlth_averages(data=filtered_data)

    # Plot GENHLTH trends
    plot_physhlth_trends(data=avg_physhlth)

    # Perform paired t-test
    pre_physhlth = matched_data[matched_data['Period'] == 'Pre-Treatment']['GENHLTH'].values
    post_physhlth = matched_data[matched_data['Period'] == 'Post-Treatment']['GENHLTH'].values

    # Calculate confidence intervals using bootstrapping
    mean_diff, conf_int = bootstrap_ci(data1=pre_physhlth, data2=post_physhlth)
    t_stat, p_value = perform_paired_ttest(pre_physhlth=pre_physhlth, post_physhlth=post_physhlth)

    # Summarize results
    summarize_results(t_stat, p_value, mean_diff, conf_int)

    print("--------------")
    print("DiD Analysis:")
    print("--------------")
    # Step 1: Filter and Prepare Data
    prepared_data = filter_and_prepare_data(data, treated_state=state, treatment_month=event_month,
                                            treatment_year=event_year, confounders=confounders)

    # Step 2: Perform DiD Analysis
    did_model = run_did_analysis(prepared_data, confounders)
    print(did_model.summary())

    # Step 3: Visualize Trends
    visualize_trends(prepared_data, event_date=pd.to_datetime(f'{event_year}-{event_month}-01'))


# Main Execution
if __name__ == "__main__":
    # Load Dataset
    file_path = 'data/processed_questionnaires_data.csv'
    questionnaires_data = pd.read_csv(file_path)

    # Run event analysis
    pipline(data=questionnaires_data, state='Connecticut', event_year=2009, event_month=1)
    print("Done!")
    pipline(data=questionnaires_data, state='Oklahoma', event_year=2006, event_month=2)
