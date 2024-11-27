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


df = load_questionnaires(['_STATE'], end_year=2023)
