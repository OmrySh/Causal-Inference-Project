import pandas as pd
import matplotlib.pyplot as plt


def create_pollution_plots():
    # Load the dataset
    data = pd.read_csv('data/pollution_data.csv')

    # Create a 'Date' column combining Year and Month
    data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))

    # List of gases and their corresponding mean and median columns
    gases = {
        "NO2": ("NO2 Mean_avg", "NO2 Mean_median"),
        "O3": ("O3 Mean_avg", "O3 Mean_median"),
        "SO2": ("SO2 Mean_avg", "SO2 Mean_median"),
        "CO": ("CO Mean_avg", "CO Mean_median")
    }

    # Create visualizations for each state
    states = data['State'].unique()

    for state in states:
        state_data = data[data['State'] == state].sort_values('Date')

        for gas, (mean_col, median_col) in gases.items():
            plt.figure(figsize=(12, 6))
            plt.plot(state_data['Date'], state_data[mean_col], label=f"{gas} Mean", linestyle='-', marker='o')
            plt.plot(state_data['Date'], state_data[median_col], label=f"{gas} Median", linestyle='--', marker='x')

            plt.title(f"{gas} Levels Over Time in {state}")
            plt.xlabel("Time")
            plt.ylabel(f"{gas} Levels")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the plot
            plt.savefig(f"figs/{state}_{gas}_levels_over_time.png")  # Save the plot as an image
            # plt.show()  # Uncomment to display the plot interactively

            plt.close()


def plot_sleep_features(file_path):
    """
    Generates time-series plots for sleep-related features for each state.

    Parameters:
    - file_path (str): Path to the dataset CSV file.

    The function will save plots for each feature (`SLEPTIME`, `ADSLEEP`, `SLEPDAY`, `SLEPDAY1`)
    aggregated by state and time (Year and Month) as PNG files.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Create a 'Date' column by combining Year and Month
    data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))

    # List of features to analyze
    features = ['SLEPTIME', 'ADSLEEP', 'SLEPDAY', 'SLEPDAY1']

    # Aggregate data by state and date
    aggregated_data = data.groupby(['State', 'Date'])[features].mean().reset_index()

    # Create visualizations for each state
    states = aggregated_data['State'].unique()

    for state in states:
        state_data = aggregated_data[aggregated_data['State'] == state]

        for feature in features:
            plt.figure(figsize=(12, 6))
            plt.plot(state_data['Date'], state_data[feature], label=f"{feature} (Avg)", linestyle='-', marker='o')

            plt.title(f"{feature} Levels Over Time in {state}")
            plt.xlabel("Time")
            plt.ylabel(f"Average {feature} Level")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the plot
            plt.savefig(f"sleepFigs/{state}_{feature}_levels_over_time.png")  # Save the plot as an image
            # plt.show()  # Uncomment to display the plot interactively

            plt.close()


def main():
    plot_sleep_features('data/questionnaires_data.csv')


if __name__ == '__main__':
    main()
