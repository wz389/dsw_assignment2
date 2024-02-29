import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def plot_average_temperature(year):
    plt.figure(figsize=(10, 6))
    subset = weather_data[weather_data['Year'] == year]
    average_temp_by_month = subset.groupby('Month')['Ftemp'].mean()

    sns.lineplot(x=average_temp_by_month.index, y=average_temp_by_month.values, marker='o')
    plt.title(f'Average Temperature by Month for Year {year}')
    plt.xlabel('Month')
    plt.ylabel('Average Temperature (Fahrenheit)')
    st.pyplot()


st.set_option('deprecation.showPyplotGlobalUse', False)

weather_data = pd.read_csv('weather.csv')

weather_data['Ftemp'] = (weather_data['Ktemp'] - 273.15) * (9/5) + 32
weather_data['Date'] = pd.to_datetime(weather_data['time'])
weather_data['Year'] = weather_data['Date'].dt.year
weather_data['Month'] = weather_data['Date'].dt.month

st.title('Part A')
st.header("Average Temperature Visualization")
selected_year = st.slider('Select Year', min_value=weather_data['Year'].min(), max_value=weather_data['Year'].max())
plot_average_temperature(selected_year)


# Part B: Find the first year when the average temperature passes 55 degrees
st.title('Part B')
first_warm_year = weather_data.groupby('Year')['Ftemp'].mean().gt(55).idxmax()
st.write(f"The first year when the average temperature passes 55 degrees is: {first_warm_year}")



# Part C: Creative data visualization
st.title('Part C')
st.header("Temperature Variation Over Seasons")

weather_data['Season'] = weather_data['Month'] % 12 // 3 + 1  # Assign each month to a season
seasonal_avg_temp = weather_data.groupby(['Year', 'Season'])['Ftemp'].mean().reset_index()

season_mapping = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
seasonal_avg_temp['Season'] = seasonal_avg_temp['Season'].map(season_mapping)

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=seasonal_avg_temp, x='Year', y='Ftemp', hue='Season', marker='o', ax=ax)
ax.set_title("Average Temperature Variation Over Seasons")
ax.set_xlabel("Year")
ax.set_ylabel("Average Temperature (Fahrenheit)")

# Short write-up
st.write("This plot illustrates the average temperature variation over seasons across different years. Each line represents a season, providing a clear visual of temperature trends throughout the years.")


st.pyplot(fig)
