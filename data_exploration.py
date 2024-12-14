
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def display_col_in_state(df, col, state):
    # df = df[df['Year'] == 2016]
    filtered = df[df['State'] == state]
    filtered = filtered[filtered[col] <= 30]

    grouped_data = (
            filtered
            .groupby(['Year', 'Month'])
            [[col]]
            .agg(['mean', 'median'])
            .reset_index()
        )
    filtered.groupby(['Year', 'Month'])[[col]].agg(['mean', 'median']).reset_index()
    # Flatten the column MultiIndex for easier access
    grouped_data.columns = ['Year', 'Month', f'{col}_mean', f'{col}_median']

    # Combine Year and Month into a single Date column for plotting
    grouped_data['Date'] = pd.to_datetime(grouped_data[['Year', 'Month']].assign(Day=1))

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(grouped_data['Date'], grouped_data[f'{col}_mean'], label=f'Mean {col}', marker='o')
    plt.plot(grouped_data['Date'], grouped_data[f'{col}_median'], label=f'Median {col}', marker='x')

    # Add labels and legend
    plt.title(f'{col} Statistics Over Time in {state}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'{col} (Hours)', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    print(grouped_data.head(37))
    print(grouped_data.tail(38))


def plot_feature_distributions(dataset, bins=20):
    """
    Plots histograms for each feature in the dataset.

    Parameters:
    - dataset: DataFrame containing the features.
    - bins: Number of bins for the histogram (default is 20).
    """
    num_features = dataset.shape[1]
    for column in ['PHYSHLTH']:
        plt.figure(figsize=(8, 5))
        plt.hist(dataset[column], bins=bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title(f"Distribution of {column}", fontsize=14)
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

processed_data = pd.read_csv('data/processed_questionnaires_data.csv')
# pre_data = pd.read_csv('data/questionnaires_data.csv')
# print(len(pre_data))
print(len(processed_data))
print(np.unique(processed_data['Year'].values))
plot_feature_distributions(processed_data)

col = 'PHYSHLTH'
states = ['Indiana', 'New Mexico', 'Kansas', 'District of Columbia', 'Connecticut', 'Maryland', 'Wyoming',
          'Nevada', 'Alabama', 'Arkansas', 'Hawaii', 'South Dakota']
states = ['Maine']
df = pd.read_csv('data/questionnaires_data.csv')

for state in states:
    display_col_in_state(df, col, state)
