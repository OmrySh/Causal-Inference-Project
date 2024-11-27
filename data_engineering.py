import numpy as np
import pandas as pd
import os


def load_questionnaires(columns, start_year=2013, end_year=2020):
    """
    Load specified columns from LLCP{year}.XPT files for a given range of years into a pandas DataFrame.

    Parameters:
    - columns (list): List of column names to extract.
    - start_year (int): Starting year for loading data. Default is 2013.
    - end_year (int): Ending year for loading data. Default is 2020.

    Returns:
    - pandas.DataFrame: DataFrame containing the specified columns from all loaded files.
    """
    all_data = []

    for year in range(start_year, end_year + 1):
        filename = f"data/questionnaires/LLCP{year}.XPT"

        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping this year.")
            continue

        try:
            # Load the file and select only the specified columns
            data = pd.read_sas(filename, format='xport', encoding='utf-8')
            filtered_data = data[columns]  # Select only the required columns
            all_data.append(filtered_data)
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            continue

    if all_data:
        # Combine all data into a single DataFrame
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    else:
        print("No data loaded. Returning an empty DataFrame.")
        return pd.DataFrame(columns=columns)


def process_pollution_data(file_path):
    """
    Process pollution data to calculate the average and median of NO2 Mean, O3 Mean, SO2 Mean, and CO Mean
    for each state and each month/year.

    Parameters:
    - file_path (str): Path to the CSV file containing pollution data.

    Returns:
    - pandas.DataFrame: A DataFrame with average and median values for each state, year, and month.
    """
    # Load the dataset
    pollution_data = pd.read_csv(file_path)

    # Convert 'Date Local' to datetime
    pollution_data['Date Local'] = pd.to_datetime(pollution_data['Date Local'])

    # Extract 'Year' and 'Month' from 'Date Local'
    pollution_data['Year'] = pollution_data['Date Local'].dt.year
    pollution_data['Month'] = pollution_data['Date Local'].dt.month

    # Group by 'State', 'Year', and 'Month' and calculate the mean and median for relevant columns
    grouped_data = (
        pollution_data
        .groupby(['State', 'Year', 'Month'])
        [['NO2 Mean', 'O3 Mean', 'SO2 Mean', 'CO Mean']]
        .agg(['mean', 'median'])
        .reset_index()
    )

    # Flatten the column MultiIndex resulting from aggregation
    grouped_data.columns = ['State', 'Year', 'Month',
                            'NO2 Mean_avg', 'NO2 Mean_median',
                            'O3 Mean_avg', 'O3 Mean_median',
                            'SO2 Mean_avg', 'SO2 Mean_median',
                            'CO Mean_avg', 'CO Mean_median']

    return grouped_data


# df = load_questionnaires(['_STATE'], end_year=2023)

file_path = 'data/uspollution_pollution_us_2000_2016.csv'
result_df = process_pollution_data(file_path)

# Display the resulting DataFrame
print(np.unique(result_df['State'].values))
print(len(np.unique(result_df['State'].values)))
