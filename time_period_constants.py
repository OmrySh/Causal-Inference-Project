import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def crate_merged_data(pollution_file, health_file):
    # Load the pollution dataset
    pollution_data = pd.read_csv(pollution_file)

    # Load the health dataset
    health_data = pd.read_csv(health_file)

    # Ensure both datasets have a consistent "Time" column
    pollution_data['Time'] = pd.to_datetime(pollution_data[['Year', 'Month']].assign(DAY=1))
    health_data['Time'] = pd.to_datetime(health_data[['Year', 'Month']].assign(DAY=1))

    # Merge datasets on State and Time
    merged_data = pd.merge(health_data, pollution_data, on=['State', 'Time'], how='inner')

    # Sort the data by State and Time
    merged_data = merged_data.sort_values(by=['State', 'Year_y', 'Month_y'])

    # Define pollutants for lagging
    pollutants = ['NO2 Mean_avg', 'O3 Mean_avg', 'SO2 Mean_avg', 'CO Mean_avg']

    # Create lagged variables
    for pollutant in pollutants:
        merged_data[f'{pollutant}_lag1'] = merged_data.groupby('State')[pollutant].shift(1)
        merged_data[f'{pollutant}_lag2'] = merged_data.groupby('State')[pollutant].shift(2)
        merged_data[f'{pollutant}_lag3'] = merged_data.groupby('State')[pollutant].shift(3)

    # Drop rows with NaN values caused by lagging
    merged_data = merged_data.dropna()

    # Inspect the data
    print(
        merged_data[['State', 'Time', 'CO Mean_avg', 'CO Mean_avg_lag1', 'CO Mean_avg_lag2', 'CO Mean_avg_lag3']].head(
            10))

    return merged_data


def process_and_visualize_lagged_data(data, health_column='PHYSHLTH'):
    """
    Processes the dataset by aggregating at the state and time level, creating lagged variables, and visualizing them.

    Parameters:
    - data (pd.DataFrame): Input dataset containing individual-level data.
    - pollutants (list of str): List of pollutant column names to process.
    - health_column (str): The column representing health outcomes (default is 'PHYSHLTH').

    Returns:
    - aggregated_data (pd.DataFrame): The aggregated and lagged dataset.
    """
    pollutants = ['NO2 Mean_avg', 'O3 Mean_avg', 'SO2 Mean_avg', 'CO Mean_avg']
    # Ensure the "Time" column exists for proper aggregation
    data['Time'] = pd.to_datetime(data[['Year_y', 'Month_y']].assign(day=1))

    # Step 1: Aggregate the data at the state and time level
    aggregated_data = data.groupby(['State', 'Year_y', 'Month_y']).agg({
        health_column: 'mean',  # Average unhealthy days
        **{pollutant: 'mean' for pollutant in pollutants}  # Average pollutant levels
    }).reset_index()

    # Rename health column for clarity
    aggregated_data.rename(columns={health_column: f'Avg_{health_column}'}, inplace=True)

    # Step 2: Sort data by State and Time
    aggregated_data = aggregated_data.sort_values(by=['State', 'Year_y', 'Month_y'])

    # Step 3: Create lagged variables
    for column in pollutants + [f'Avg_{health_column}']:
        aggregated_data[f'{column}_lag1'] = aggregated_data.groupby('State')[column].shift(1)
        aggregated_data[f'{column}_lag2'] = aggregated_data.groupby('State')[column].shift(2)
        aggregated_data[f'{column}_lag3'] = aggregated_data.groupby('State')[column].shift(3)

    # Drop rows with NaN values caused by lagging
    aggregated_data = aggregated_data.dropna()

    # Step 4: Visualization Function
    def visualize_lagged_variables(data, column):
        plt.figure(figsize=(12, 6))
        sns.histplot(data[column], kde=True, color='blue', label='Current', bins=30, alpha=0.6)
        sns.histplot(data[f'{column}_lag1'], kde=True, color='orange', label='Lag 1 Month', bins=30, alpha=0.6)
        sns.histplot(data[f'{column}_lag2'], kde=True, color='green', label='Lag 2 Months', bins=30, alpha=0.6)
        sns.histplot(data[f'{column}_lag3'], kde=True, color='red', label='Lag 3 Months', bins=30, alpha=0.6)
        plt.title(f'Distribution of {column} and Its Lags')
        plt.xlabel(f'{column} Levels')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    # Step 5: Call visualization for each pollutant and health column
    for column in pollutants + [f'Avg_{health_column}']:
        visualize_lagged_variables(data=aggregated_data, column=column)

    aggregated_data.to_csv('aggregated_lagged_data_fixed.csv', index=False)


if __name__ == '__main__':
    pollution_file_path = 'data/pollution_data.csv'  # Replace with your actual file path
    health_file_path = 'data/questionnaires_data.csv'  # Replace with your actual file path
    merged_data_time = crate_merged_data(pollution_file=pollution_file_path, health_file=health_file_path)
    process_and_visualize_lagged_data(data=merged_data_time)
