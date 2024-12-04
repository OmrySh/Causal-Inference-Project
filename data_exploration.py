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


col = 'PHYSHLTH'
states = ['Indiana', 'New Mexico', 'Kansas', 'District of Columbia', 'Connecticut', 'Maryland', 'Wyoming',
          'Nevada', 'Alabama', 'Arkansas', 'Hawaii', 'South Dakota']
states = ['Maine']
df = pd.read_csv('data/questionnaires_data.csv')

for state in states:
    display_col_in_state(df, col, state)
