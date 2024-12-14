import numpy as np
import pandas as pd
import os
import time
import pickle

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



def load_questionnaires_with_mapping(column_mapping={}, start_year=2010, end_year=2020):
    """
    Load XPT files and handle inconsistent column names, selecting required columns.

    Parameters:
    - start_year (int): Starting year for loading data. Default is 2010.
    - end_year (int): Ending year for loading data. Default is 2020.

    Returns:
    - pandas.DataFrame: Unified DataFrame with consistent column names.
    """
    # Column mappings to handle inconsistent naming across years
    column_mapping['_STATE'] = ['_STATE']  # State number
    column_mapping['IMONTH']= ['IMONTH']  # Interview month
    column_mapping['IYEAR']= ['IYEAR']  # Interview year
    column_mapping['SLEPTIME']= ['SLEPTIME', 'SLEPTIM1']  # Hours of sleep
        # 'ADSLEEP': ['ADSLEEP'],  # Trouble sleeping (2012-2014 missing)
        # 'SLEPDAY': ['SLEPDAY'],  # Falling asleep during day (2010-2012 only)
        # 'SLEPDAY1': ['SLEPDAY1']  # Falling asleep during day (2016-2018 only)
    # con_founders_columns = pd.read_csv('data/con_founders_questions.csv')
    # for con_founder in con_founders_columns['code'].values:
    #     column_mapping[con_founder] = con_founder

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


def process_features(dataframe, dataset):
    """
    Processes a dataset based on feature rules provided in the dataframe.

    Parameters:
    - dataframe: DataFrame with feature definitions, ranges, and notes.
    - dataset: DataFrame with the actual dataset to process.

    Returns:
    - Processed dataset.
    """
    processed_dataset = dataset.copy()

    for _, row in dataframe.iterrows():
        feature = row['feature']
        low = row['low']
        high = row['high']
        notes = row['notes']
        print(feature)
        print("before", len(processed_dataset))
        if not pd.isna(notes) and 'None is' in notes:
            replace_value = int(notes.split(' ')[-1])
            if replace_value < low:
                low = replace_value
            processed_dataset[feature].fillna(replace_value, inplace=True)
            if "88" not in notes:
                processed_dataset = processed_dataset[(processed_dataset[feature] >= low) & (processed_dataset[feature] <= high)]

        if pd.isna(notes):
            # Keep only rows within range [low, high]
            processed_dataset = processed_dataset[(processed_dataset[feature] >= low) & (processed_dataset[feature] <= high)]

        elif "88" in notes:
            # Replace 88 with 0 and keep rows within range [low, high] or equal to 0
            processed_dataset[feature] = processed_dataset[feature].apply(lambda x: 0 if x == 88 else x)
            processed_dataset = processed_dataset[(processed_dataset[feature].between(low, high)) | (processed_dataset[feature] == 0)]

        elif notes == "one hot":
            # Convert to one-hot encoding; keep only rows in range
            valid_rows = (processed_dataset[feature] >= low) & (processed_dataset[feature] <= high)
            one_hot = pd.get_dummies(processed_dataset.loc[valid_rows, feature], prefix=feature).astype(int)
            processed_dataset = processed_dataset[valid_rows].drop(columns=[feature])
            processed_dataset = pd.concat([processed_dataset, one_hot], axis=1)


        print("after", len(processed_dataset))


    cols_to_drop = []
    for col in processed_dataset.keys():
        if col not in dataframe['feature'].values and col not in ['State', 'Year', 'Month'] and 'MARITAL' not in col:
            cols_to_drop.append(col)
    processed_dataset = processed_dataset.drop(columns=cols_to_drop)

    return processed_dataset

def find_common_features(features_dict):
    # Start with the features from the first year
    common_features = set(next(iter(features_dict.values())))

    # Intersect with features from all other years
    for features in features_dict.values():
        common_features.intersection_update(features)

    # Convert the result back to a list
    return list(common_features)

def get_common_features_dict():
    with open('data/features.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    print(loaded_data)
    features_by_year = {}
    for i in range(2005, 2019):
        features_by_year[i] = loaded_data[f'{i}_yes']

    common_features = find_common_features(features_by_year)
    common_dict = {}
    for common in common_features:
        common_dict[common] = [common]
    print(common_features)

def find_features_in_questionnaires():
    count_dic = {}
    q = pd.read_csv('data/con_founders_questions.csv')
    features_dict = {}
    for i in range(2005, 2019):
        features_dict[f'{i}_yes'] = []
        features_dict[f'{i}_no'] = []
        df = pd.read_sas(f'data/questionnaires/LLCP{i}.XPT')
        count = 0
        for c in q['code'].values:
            if c in df.keys():
                print(f"{c} yes")
                features_dict[f'{i}_yes'].append(c)
                count += 1
            else:
                print(f"{c} no")
                features_dict[f'{i}_no'].append(c)

        count_dic[i] = count

    with open('data/features.pkl', 'wb') as file:
        pickle.dump(features_dict, file)
    for year in count_dic:
        print(f'{year}: {count_dic[year]} out of {len(q["code"].values)}')

def run_pre_processing(common_dict):
    questionnaires_df = load_questionnaires_with_mapping(column_mapping=common_dict,start_year=2005, end_year=2018)
    questionnaires_df.to_csv('data/questionnaires_data.csv')
    # pollution_path = 'data/uspollution_pollution_us_2000_2016.csv'
    # pollution_df = process_pollution_data(pollution_path)
    # pollution_df.to_csv('data/pollution_data.csv')
    #
    # merged_df = merge_pollution_and_xpt(pollution_df, questionnaires_df)
    # merged_df.to_csv('data/merged_data.csv')


features_range = pd.read_csv('data/features_value_range.csv')
for note in features_range['notes'].values:
    print(note)
questionnaires_dataset = pd.read_csv('data/questionnaires_data.csv')
processed_dataset = process_features(features_range, questionnaires_dataset)
processed_dataset.to_csv('data/processed_questionnaires_data.csv')

