from django.shortcuts import render
from django.http import HttpResponse
from .models import TemperaturePrediction
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def index(request):
    return render(request, 'temperature_app/index.html')

def predict_temperature(request):
    if request.method == 'POST':
        user_city = request.POST.get('city')
        user_year = int(request.POST.get('year'))
        user_month = int(request.POST.get('month'))

        # Problem Statement:
        # The goal of this project is to analyze global and cities temperature data to understand long-term trends.
        # Specifically, we aim to build predictive models for land average temperatures and derive actionable insights.

        # Load the original dataset
        file_path_global = 'GlobalTemperatures.csv'
        df_global = pd.read_csv(file_path_global)

        # Display the first few rows of the original dataset
        print(df_global.head())

        # Overview of the original dataset
        print(df_global.info())

        # Data Cleaning for the original dataset
        # Drop rows with null values in the target variable
        df_global.dropna(subset=['LandAndOceanAverageTemperature'], inplace=True)

        # Feature engineering: Extract year and month from the date
        df_global['dt'] = pd.to_datetime(df_global['dt'])
        df_global['Year'] = df_global['dt'].dt.year
        df_global['Month'] = df_global['dt'].dt.month

        # Statistical Summaries for the original dataset
        print("Statistical Summaries:")
        print(df_global.describe())

        # Enhanced Exploratory Data Analysis (EDA)
        # Visualize temperature distributions
        plt.figure(figsize=(12, 6))
        sns.histplot(df_global['LandAndOceanAverageTemperature'], bins=30, kde=True)
        plt.title('Distribution of Land and Ocean Average Temperature')
        plt.xlabel('Land and Ocean Average Temperature')
        plt.ylabel('Frequency')
        plt.show()

        # Explore seasonality in temperature data
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Month', y='LandAndOceanAverageTemperature', data=df_global)
        plt.title('Seasonal Patterns in Land and Ocean Average Temperature')
        plt.xlabel('Month')
        plt.ylabel('Land and Ocean Average Temperature')
        plt.show()

        # Use correlation matrices to identify relationships between variables
        correlation_matrix = df_global.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
        plt.title('Correlation Matrix')
        plt.show()

        print("***************Land and Ocean Average Temperature****************")

        # Select relevant features for modeling for the original dataset
        features_global = ['Year', 'Month']
        X_global = df_global[features_global]
        y_global = df_global['LandAndOceanAverageTemperature']

        # Split the dataset into training and testing sets for the original dataset
        X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
            X_global, y_global, test_size=0.2, random_state=42
        )

        # Model Training for the original dataset
        # Create a linear regression model
        model_global = LinearRegression()

        # Train the model
        model_global.fit(X_train_global, y_train_global)

        # Make predictions on the test set
        y_pred_global = model_global.predict(X_test_global)

        # Evaluate the model
        mse_global = mean_squared_error(y_test_global, y_pred_global)
        print(f'Mean Squared Error for Global Dataset: {mse_global}')

        # Visualization for the original dataset
        # Visualize predicted vs actual values
        plt.scatter(X_test_global['Year'], y_test_global, label='Actual')
        plt.scatter(X_test_global['Year'], y_pred_global, color='red', label='Predicted')
        plt.xlabel('Year')
        plt.ylabel('Land and Ocean Average Temperature')
        plt.legend()
        plt.title('Linear Regression: Predicted vs Actual Land and Ocean Average Temperature (Global)')
        plt.show()

        # Load the new dataset "GlobalTemperatureofcities"
        file_path_cities = 'GlobalLandTemperaturesByCityFiltered.csv'
        df_cities = pd.read_csv(file_path_cities)

        # Display the first few rows of the new dataset
        print(df_cities.head())

        # Overview of the new dataset
        print(df_cities.info())

        # Data Cleaning for the new dataset
        # Drop rows with null values in the target variable
        df_cities.dropna(subset=['AverageTemperature'], inplace=True)

        # Feature engineering: Extract year and month from the date for df_cities
        df_cities['dt'] = pd.to_datetime(df_cities['dt'], format='%m/%d/%Y', errors='coerce')
        df_cities['Year'] = df_cities['dt'].dt.year
        df_cities['Month'] = df_cities['dt'].dt.month

        print(df_cities['dt'].isnull())

        # Statistical Summaries for the new dataset
        print("Statistical Summaries for cities:")
        print(df_cities.describe())

        # Enhanced Exploratory Data Analysis (EDA) for the new dataset
        # Time Series Analysis
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='dt', y='AverageTemperature', data=df_cities)
        plt.title('cities Average Temperature Over Time')
        plt.show()


        print("***************cities Average Temperature****************")

        # Select relevant features for modeling for the new dataset
        features_cities = ['Year', 'Month']
        X_cities = df_cities[features_cities]
        y_cities = df_cities['AverageTemperature']

        # Split the dataset into training and testing sets for the new dataset
        X_train_cities, X_test_cities, y_train_cities, y_test_cities = train_test_split(
            X_cities, y_cities, test_size=0.2, random_state=42
        )

        # Drop rows with missing values jointly for features and target
        Xy_train_cities = pd.concat([X_train_cities, y_train_cities], axis=1).dropna()
        X_train_cities = Xy_train_cities[features_cities]
        y_train_cities = Xy_train_cities['AverageTemperature']

        Xy_test_cities = pd.concat([X_test_cities, y_test_cities], axis=1).dropna()
        X_test_cities = Xy_test_cities[features_cities]
        y_test_cities = Xy_test_cities['AverageTemperature']

        # Model Training for the new dataset
        # Create a linear regression model
        model_cities = LinearRegression()

        # Train the model
        model_cities.fit(X_train_cities, y_train_cities)

        # Make predictions on the test set
        y_pred_cities = model_cities.predict(X_test_cities)

        # Evaluate the model
        mse_cities = mean_squared_error(y_test_cities, y_pred_cities)
        print(f'Mean Squared Error for cities Dataset: {mse_cities}')

        # Visualization for the new dataset
        # Visualize predicted vs actual values
        plt.scatter(X_test_cities['Year'], y_test_cities, label='Actual')
        plt.scatter(X_test_cities['Year'], y_pred_cities, color='red', label='Predicted')
        plt.xlabel('Year')
        plt.ylabel('Average Temperature (cities)')
        plt.legend()
        plt.title('Linear Regression: Predicted vs Actual Average Temperature (cities)')
        plt.show()

        # Feature engineering: Extract year and month from the date for df_global
        df_global['dt'] = pd.to_datetime(df_global['dt'], format='%Y-%m-%d')
        df_global['Year'] = df_global['dt'].dt.year
        df_global['Month'] = df_global['dt'].dt.month



        # Join the datasets on the 'dt' column
        merged_df = pd.merge(df_global, df_cities, on='dt', how='inner')
        output_file_path = 'CombinedTemperatureData.csv'
        merged_df.to_csv(output_file_path, index=False)

        # Read the dataset
        file_path_global = 'CombinedTemperatureData.csv'
        df = pd.read_csv(file_path_global)

        # Convert 'Latitude' and 'Longitude' to numeric values
        df['Latitude'] = pd.to_numeric(df['Latitude'].str.rstrip('NS'))
        df['Longitude'] = pd.to_numeric(df['Longitude'].str.rstrip('EW'))

        # Rename columns
        df.rename(columns={'Year_x': 'Year_global', 'Month_x': 'Month_global'}, inplace=True)

        # Select relevant features for temperature modeling
        features_temperature = ['Latitude', 'Longitude', 'Year_global', 'Month_global']
        X_temperature = df[features_temperature]
        y_max_temperature = df['LandMaxTemperature']
        y_min_temperature = df['LandMinTemperature']

        # Select relevant features for city prediction
        features_city = ['Latitude', 'Longitude', 'Year_global', 'Month_global']
        X_city = df[features_city]
        y_city = df['City']

        # Create models for temperature prediction
        model_max_temperature = LinearRegression()
        model_min_temperature = LinearRegression()

        # Create a model for city prediction (using KNeighborsClassifier)
        model_city = KNeighborsClassifier(n_neighbors=5)

        # Train the models for temperature prediction
        model_max_temperature.fit(X_temperature, y_max_temperature)
        model_min_temperature.fit(X_temperature, y_min_temperature)

        # Train the model for city prediction
        model_city.fit(X_city, y_city)

        # Assuming you have Latitude and Longitude information for the user-provided city
        user_city_data = df.loc[(df['City'] == user_city) & (df['Month_global'] == user_month), 
                                ['Latitude', 'Longitude', 'Year_global', 'Month_global']]

        # Include the user input month and year in the input features
        user_city_data.loc[:, 'Year_global'] = user_year

        if user_city_data.shape[0] > 0:
            # Make predictions for temperature
            temperature_prediction_max = model_max_temperature.predict(user_city_data[features_temperature])
            temperature_prediction_min = model_min_temperature.predict(user_city_data[features_temperature])

            # Make prediction for city
            city_prediction = model_city.predict(user_city_data[features_city])

            print(f'Predicted LandMaxTemperature for {user_city} in {user_month}/{user_year}: {temperature_prediction_max[0]}')
            print(f'Predicted LandMinTemperature for {user_city} in {user_month}/{user_year}: {temperature_prediction_min[0]}')
            print(f'Predicted City for {user_city} in {user_month}/{user_year}: {city_prediction[0]}')
        else:
            print("No data available for the specified city, month, and year.")
        
        context = {
                'user_city': user_city,
                'user_year': user_year,
                'user_month': user_month,
                'temperature_prediction_max': temperature_prediction_max[0],
                'temperature_prediction_min': temperature_prediction_min[0],
                'city_prediction': city_prediction[0],
            }

        TemperaturePrediction.objects.create(
                city=user_city,
                year=user_year,
                month=user_month,
                max_temperature=temperature_prediction_max[0],
                min_temperature=temperature_prediction_min[0],
                city_prediction=city_prediction[0]
            )

        return render(request, 'temperature_app/prediction_result.html', context)
    return HttpResponse("Invalid request method.")
