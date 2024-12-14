import pandas as pd
import plotly.express as px


def visUSA_health():
    # Load the dataset
    file_path = 'data/questionnaires_data_health.csv'  # Replace with your actual file path
    data = pd.read_csv(file_path)

    # Replace specific values in the PHYSHLTH column
    data['PHYSHLTH'] = data['PHYSHLTH'].replace({
        88: 0,  # Replace 'None' with 0 (no unhealthy days)
        77: None,  # Replace 'Don’t know/Not sure' with NaN
        99: None,  # Replace 'Refused' with NaN
    })

    # Drop rows with invalid or missing PHYSHLTH values
    data = data.dropna(subset=['PHYSHLTH'])

    # Keep only valid range (1–30) and 0 for "None"
    data = data[(data['PHYSHLTH'] >= 0) & (data['PHYSHLTH'] <= 30)]

    # Aggregate the data: average PHYSHLTH by State, Year, and Month
    aggregated_data = data.groupby(['State', 'Year', 'Month'])['PHYSHLTH'].mean().reset_index()

    # Create a new column for time (Year-Month)
    aggregated_data['Time'] = pd.to_datetime(aggregated_data[['Year', 'Month']].assign(DAY=1))
    aggregated_data['Time'] = aggregated_data['Time'].dt.strftime('%Y-%m')

    # Define a mapping of state names to two-letter abbreviations
    state_abbreviations = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
        'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
        'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
        'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
        'Pennsylvania': 'PA', 'Puerto Rico': None, 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virgin Islands': None, 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
        'Wisconsin': 'WI', 'Wyoming': 'WY'
    }

    # Map state names to abbreviations
    aggregated_data['State'] = aggregated_data['State'].map(state_abbreviations)

    # Drop rows where the state could not be mapped
    aggregated_data = aggregated_data.dropna(subset=['State'])

    # Determine the global min and max values for PHYSHLTH across all time periods
    physhlth_min = aggregated_data['PHYSHLTH'].min()
    physhlth_max = aggregated_data['PHYSHLTH'].max()

    # Plot a choropleth map with a uniform color scale
    fig = px.choropleth(
        aggregated_data,
        locations='State',  # Two-letter abbreviations
        locationmode='USA-states',
        color='PHYSHLTH',  # Color represents the average number of unhealthy days
        hover_name='State',  # State name appears on hover
        hover_data={'Time': True, 'PHYSHLTH': True},  # Include time and PHYSHLTH in hover info
        animation_frame='Time',  # Add a slider for time
        color_continuous_scale='Reds',  # Darker red = more unhealthy days
        range_color=[physhlth_min, physhlth_max],  # Set a uniform color scale
        title='Average Number of Unhealthy Days per State Over Time',
        scope='usa'  # Limit map to the United States
    )

    # Update layout for better readability
    fig.update_layout(
        geo=dict(scope='usa'),
        coloraxis_colorbar=dict(title='Avg Unhealthy Days'),
        title_x=0.5
    )

    # Show the interactive plot
    fig.show()


if __name__ == '__main__':
    visUSA_health()
