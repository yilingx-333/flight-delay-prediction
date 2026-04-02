# Flight Delay Prediction
This project aims to predict flight arrival delays for domestic U.S. flights using machine learning methods.
The project integrates multiple data sources, including flight operational data, airport geographic data, and weather data obtained from an external API.
The workflow includes data collection, preprocessing, feature engineering, exploratory data analysis (EDA), and model development using Linear Regression, Random Forest, and XGBoost.


# Data sources
1. U.S. DOT On-Time Performance Data
Source: https://www.transtats.bts.gov/
Type: CSV file
Description: Contains flight-level data such as departure time, arrival time, delays, carrier, origin, destination, and distance.
Size: ~35,000 records (January-March 2014 after filtering)

2. NOAA Weather API
Source: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
Type: API (JSON)
Description: Provides daily weather data including maximum temperature (TMAX) and precipitation (PRCP).
Data is retrieved programmatically and merged with flight data by date.

3. OpenFlights Airport Database
Source: https://openflights.org/data.html
Type: CSV file
Description: Provides airport information including IATA code and geographic coordinates.
Used to enrich the dataset and compute great-circle distance.

# Results 
The project compares three machine learning models:
Model               	MAE      	RMSE        	R²
Linear Regression   	~8.31	    ~11.44	    ~0.84
Random Forest       	~7.46	    ~10.27	    ~0.87
XGBoost	                ~7.04	    ~9.81	    ~0.88

Departure delay is the strongest predictor of arrival delay.
Nonlinear models (Random Forest, XGBoost) outperform Linear Regression.
XGBoost achieves the best performance, indicating strong capability in capturing complex patterns.
Weather variables have limited but noticeable impact on delays.

# Installation
NOAA API key:
Create a file named `.env` in the project root directory:
NOAA_API_KEY=your_api_key_here
You can also copy `.env.example` and rename it to `.env`:
`cp .env.example .env`

## Required Python Packages

Install required packages using:

Main dependencies:
pandas
numpy
matplotlib
scikit-learn
xgboost
requests

# Running analysis 
Data Preparation  
`python src/main.py`
Load and clean data,Fetch weather data from NOAA API,Generate final_dataset.csv

Exploratory Data Analysis (EDA)
`python src/analyze.py`
Generate plots,Save outputs to results/eda/

Model Training
Baseline model:
`python src/baseline.py`
Random Forest:
`python src/random_forest.py`
XGBoost:
`python src/XGBoost.py`

# Output
All results are saved in:
results/
    final_dataset.csv
    eda/
    model_outputs/