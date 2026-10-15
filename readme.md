# Olist Delivery Time Prediction - Machine Learning

This project uses machine learning to predict delivery times for orders placed on Olist, a Brazilian e-commerce platform.

## Overview

The goal of this project is to build a predictive model that estimates the time it takes for an order to reach the customer. This can help Olist provide better delivery estimates and improve customer satisfaction.

## Data

The project uses the "Brazilian E-Commerce Public Dataset by Olist," available on Kaggle. The dataset contains information on 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil.

It includes features such as:
*   Order details (status, purchase timestamp, etc.)
*   Customer information (location, etc.)
*   Product information (category, weight, dimensions, etc.)
*   Payment and review details
*   Geolocation data

## Methodology

The project follows a standard machine learning workflow:
1.  **Data Ingestion:** Loading multiple datasets using `pandas`.
2.  **Data Cleaning & Preprocessing:** Handling missing values, converting timestamps, and merging datasets.
3.  **Geospatial Analysis:** Utilizing `duckdb` with spatial extensions to process geolocation data.
4.  **Feature Engineering:** Creating new features relevant to delivery time prediction.
5.  **Modeling:** Training and evaluating various regression models (Linear Regression, Ridge, Lasso, ElasticNet, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, XGBRegressor, LGBMRegressor).

## Requirements

To run the notebook, you will need the following libraries:
*   pandas
*   python-dotenv
*   tqdm
*   duckdb
*   numpy
*   seaborn
*   matplotlib
*   scipy
*   statsmodels
*   scikit-learn
*   xgboost
*   lightgbm

## Usage

1.  Clone this repository.
2.  Ensure you have the required libraries installed.
3.  Set up your `.env` file with the correct data paths.
4.  Run the Jupyter Notebook to explore the data, train the models, and view the results.