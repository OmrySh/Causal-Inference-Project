import numpy as np
import pandas as pd
import os
import time


def count_numerical_sleeptime(df, col):
    """
    Group the DataFrame by _STATE and count non-NaN numerical values in the SLEPTIME column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing _STATE and SLEPTIME columns.

    Returns:
    - pd.DataFrame: A DataFrame with _STATE and the count of valid numerical values in SLEPTIME.
    """
    # Ensure SLEPTIME contains only numerical values and drop non-numeric rows
    df['SLEPTIME_numeric'] = pd.to_numeric(df[col], errors='coerce')

    # Group by _STATE and count non-NaN numerical values in SLEPTIME
    result = (
        df.groupby('_STATE')['SLEPTIME_numeric']
        .apply(lambda x: x.notna().sum())
        .reset_index(name='Valid_SLEPTIME_Count')
    )

    for row in result.iterrows():
        print(row)

    return result


def load_questionnaires_with_mapping(start_year=2010, end_year=2020):
    """
    Load XPT files and handle inconsistent column names, selecting required columns.

    Parameters:
    - start_year (int): Starting year for loading data. Default is 2010.
    - end_year (int): Ending year for loading data. Default is 2020.

    Returns:
    - pandas.DataFrame: Unified DataFrame with consistent column names.
    """
    # Column mappings to handle inconsistent naming across years
    column_mapping = {
        '_STATE': ['_STATE'],  # State number
        'IMONTH': ['IMONTH'],  # Interview month
        'IYEAR': ['IYEAR'],  # Interview year
        'SLEPTIME': ['SLEPTIME', 'SLEPTIM1'],  # Hours of sleep
        'PHYSHLTH': ['PHYSHLTH']
        # 'ADSLEEP': ['ADSLEEP'],  # Trouble sleeping (2012-2014 missing)
        # 'SLEPDAY': ['SLEPDAY'],  # Falling asleep during day (2010-2012 only)
        # 'SLEPDAY1': ['SLEPDAY1']  # Falling asleep during day (2016-2018 only)
    }

    all_data = []

    for year in range(start_year, end_year + 1):
        filename = f"data/questionnaires/LLCP{year}.XPT"
        print("Processing:", filename)

        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping this year.")
            continue

        try:
            # Load the data
            data = pd.read_sas(filename)

            # Initialize a dictionary for selected columns
            selected_columns = {}
            missing_columns = []

            for target_col, possible_names in column_mapping.items():
                # Find the first available column name in the dataset
                for col in possible_names:
                    if col in data.columns:
                        selected_columns[target_col] = col
                        break
                else:
                    # If no column name is found, mark it as missing
                    missing_columns.append(target_col)

            # Select and rename columns
            data_subset = data[list(selected_columns.values())].rename(
                columns={v: k for k, v in selected_columns.items()})

            # Add missing columns as NaN
            for col in missing_columns:
                print(f"Column '{col}' not found in any of the possible names for year {year}. Setting it to NaN.")
                data_subset[col] = pd.NA
            if 'SLEPTIME' in data_subset.keys():
                count_numerical_sleeptime(data_subset, 'SLEPTIME')
            elif 'SLEPTIM1' in data_subset.keys():
                count_numerical_sleeptime(data_subset, 'SLEPTIME')
            data_subset['Year'] = year  # Add year column
            all_data.append(data_subset)

        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            continue

    if all_data:
        # Combine all data into a single DataFrame
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = convert_date_bytes(combined_data)
        combined_data = convert_state_codes(combined_data)
        return combined_data
    else:
        print("No data loaded. Returning an empty DataFrame.")
        return pd.DataFrame(columns=column_mapping.keys())


def convert_date_bytes(df):
    # Convert IMONTH and IYEAR columns from bytes to int and rename them
    df['Month'] = df['IMONTH'].apply(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else x)
    df['Year'] = df['IYEAR'].apply(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else x)

    # Drop the original byte columns
    df = df.drop(columns=['IMONTH', 'IYEAR'])
    return df


def convert_state_codes(df, codebook_path='data/states_codebook.txt'):
    """
    Convert state codes in the _STATE column to state names using a codebook file.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the _STATE column with state codes.
    - codebook_path (str): Path to the text file containing the state codebook.

    Returns:
    - pd.DataFrame: DataFrame with an additional 'State' column containing state names.
    """
    # Load the codebook file into a dictionary
    try:
        with open(codebook_path, 'r') as f:
            codebook = {
                int(line.split()[0]): ' '.join(line.split()[1:]) for line in f.readlines()
            }
    except Exception as e:
        raise FileNotFoundError(f"Error loading the codebook file: {e}")

    # Map the _STATE column to state names
    df['State'] = df['_STATE'].map(codebook)

    # Return the DataFrame with the new 'State' column
    return df


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

def merge_pollution_and_xpt(pollution_df, questionnaires_df):
    """
    Merge pollution data and XPT result data based on the Year and Month columns.

    Parameters:
    - pollution_df (pd.DataFrame): DataFrame containing pollution data with 'Year' and 'Month' columns.
    - xpt_result_df (pd.DataFrame): DataFrame containing XPT survey data with 'Year' and 'Month' columns.

    Returns:
    - pd.DataFrame: Merged DataFrame containing data from both input DataFrames.
    """
    # Ensure the Year and Month columns exist and are of the same type in both DataFrames
    pollution_df['Year'] = pollution_df['Year'].astype(int)
    pollution_df['Month'] = pollution_df['Month'].astype(int)

    questionnaires_df['Year'] = questionnaires_df['Year'].astype(int)
    questionnaires_df['Month'] = questionnaires_df['Month'].astype(int)

    # Perform the merge
    merged_df = pd.merge(pollution_df, questionnaires_df, on=['Year', 'Month', 'State'], how='inner')

    return merged_df
"""
Code Book
SLEPTIME or SLEPTIM1: On average, how many hours of sleep do you get in a 24-hour period?
All year except 2015
ADSLEEP: Over the last 2 weeks, how many days have you had trouble falling asleep or staying asleep or sleeping too much?
All except 2012-2014, 2016
SLEPDAY: During the past 30 days, for about how many days did you find yourself unintentionally falling asleep during the day?
Only in 2010-2012
SLEPDAY1: Over the last 2 weeks, how many days did you unintentionally fall asleep during the day?
Only in 2017-2018
"""


def run_pre_processing():
    questionnaires_df = load_questionnaires_with_mapping(start_year=2005, end_year=2018)
    questionnaires_df.to_csv('data/questionnaires_data.csv')
    pollution_path = 'data/uspollution_pollution_us_2000_2016.csv'
    pollution_df = process_pollution_data(pollution_path)
    pollution_df.to_csv('data/pollution_data.csv')

    merged_df = merge_pollution_and_xpt(pollution_df, questionnaires_df)
    merged_df.to_csv('data/merged_data.csv')


run_pre_processing()
