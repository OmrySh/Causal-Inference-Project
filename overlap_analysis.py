import pandas as pd
import os
import scipy.stats as stats
import matplotlib.pyplot as plt

def distribution_analysis_by_date_and_state(dataset, features, month, year, state):
    """
    Analyzes and compares feature distributions before and after a given date
    for a specific state and other states, and saves plots.

    Parameters:
    - dataset: DataFrame containing features and "Month", "Year", and "State" columns.
    - features: List of features to analyze.
    - month: Target month (integer).
    - year: Target year (integer).
    - state: The specific state to analyze (string).

    Returns:
    - A dictionary with distribution statistics and visualizations for each feature.
    """
    # Ensure the output directory exists
    output_dir = "data/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Calculate date boundaries
    before_start_date = pd.Timestamp(year=year, month=month, day=1) - pd.DateOffset(months=11)
    before_end_date = pd.Timestamp(year=year, month=month, day=1) - pd.DateOffset(months=1)
    after_start_date = pd.Timestamp(year=year, month=month, day=1)
    after_end_date = pd.Timestamp(year=year, month=month, day=1) + pd.DateOffset(months=11)

    # Create date column for filtering
    dataset['Date'] = pd.to_datetime(dataset[['Year', 'Month']].assign(day=1))

    # Filter data for the two periods
    before_data = dataset[(dataset['Date'] >= before_start_date) & (dataset['Date'] <= before_end_date)]
    after_data = dataset[(dataset['Date'] >= after_start_date) & (dataset['Date'] <= after_end_date)]

    # Further filter by state
    state_data_before = before_data[before_data['State'] == state]
    state_data_after = after_data[after_data['State'] == state]
    other_states_before = before_data[before_data['State'] != state]
    other_states_after = after_data[after_data['State'] != state]

    # Analyze distributions
    analysis_results = {}
    for feature in features:
        state_before = state_data_before[feature].dropna()
        state_after = state_data_after[feature].dropna()
        other_before = other_states_before[feature].dropna()
        other_after = other_states_after[feature].dropna()

        # Compute statistics
        state_stats = {
            "Before Mean": state_before.mean(),
            "After Mean": state_after.mean(),
            "Before Std": state_before.std(),
            "After Std": state_after.std(),
            "KS Test P-Value": stats.ks_2samp(state_before, state_after).pvalue
        }

        other_stats = {
            "Before Mean": other_before.mean(),
            "After Mean": other_after.mean(),
            "Before Std": other_before.std(),
            "After Std": other_after.std(),
            "KS Test P-Value": stats.ks_2samp(other_before, other_after).pvalue
        }

        # Store results
        analysis_results[feature] = {
            "State Stats": state_stats,
            "Other States Stats": other_stats
        }

        # Plot and save for the state
        plt.figure(figsize=(10, 6))
        plt.hist(state_before, bins=20, alpha=0.5, label='State Before', color='blue', density=True)
        plt.hist(state_after, bins=20, alpha=0.5, label='State After', color='orange', density=True)
        plt.title(f"Distribution Comparison for {feature} (State: {state})", fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        state_plot_path = os.path.join(output_dir, f"{feature}_{state}.png")
        plt.savefig(state_plot_path)
        plt.close()

        # Plot and save for other states
        plt.figure(figsize=(10, 6))
        plt.hist(other_before, bins=20, alpha=0.5, label='Other States Before', color='green', density=True)
        plt.hist(other_after, bins=20, alpha=0.5, label='Other States After', color='red', density=True)
        plt.title(f"Distribution Comparison for {feature} (Other States)", fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        other_states_plot_path = os.path.join(output_dir, f"{feature}_other_states.png")
        plt.savefig(other_states_plot_path)
        plt.close()

    return analysis_results



dataset = pd.read_csv('data/processed_questionnaires_data.csv')
features = [
    "GENHLTH", "POORHLTH", "SMOKDAY2", "WEIGHT2", "EXERANY2", "SMOKE100",
    "MENTHLTH", "EDUCA", "HEIGHT3", "PREGNANT", "CHILDREN",
    "MARITAL_1.0", "MARITAL_2.0", "MARITAL_3.0", "MARITAL_4.0",
    "MARITAL_5.0", "MARITAL_6.0"
]
month = 2
year = 2006
state = 'Indiana'
results = distribution_analysis_by_date_and_state(dataset, features, month, year, state)
print(results)
