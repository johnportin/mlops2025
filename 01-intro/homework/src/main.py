from typing import List
import pandas as pd
import sklearn
import os
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

dv = DictVectorizer()

def ingest_parquet_data(filename: str) -> pd.DataFrame:
    if not os.path.exists(filename):
        raise FileNotFoundError
    return pd.read_parquet(filename)

def drop_outliers(data):
    data['datetime_difference'] = data.tpep_dropoff_datetime - data.tpep_pickup_datetime
    data['duration'] = data['datetime_difference'].astype('int64') / ( 60 * 10 ** 6)
    data = data[(data['duration'] <= 60) & (data['duration'] >= 1)]
    total_rows_after_dropping_outliers = data.shape[0]
    print(f"Cols before: {total_rows}, cols after: {total_rows_after_dropping_outliers}")
    print(f"Percent remaining: {100 * total_rows_after_dropping_outliers/total_rows:.2f}%")
    return data

def process_data(data, columns: List[str]):
    print(f"Casting categorical variables {columns} as strings...")
    data[columns] = data[columns].astype(str)

    print(f"Creating train dicts. for {[columns]}..")
    train_dicts = data[columns].to_dict(orient='records')

    x_train = dv.transform(train_dicts)
    print(f"{x_train.shape=}")

    return x_train

if __name__=="__main__":
    # Read in data
    yellow_taxi_data = ingest_parquet_data('01-intro/homework/resources/yellow_tripdata_2023-01.parquet')

    q1 = """
Question 1:
Read the data for January. How many columns are there?
Download the data for January and February 2023."""
    yellow_taxi_data_columns = yellow_taxi_data.columns
    print(q1)
    print(f"Yellow taxi data contains {len(yellow_taxi_data_columns)} columns.")
    print(f"The columns for the yellow taxi data are:")
    print(f"{yellow_taxi_data.shape=}")
    for col in yellow_taxi_data_columns:
        print(f"\t{col}")

    q2 = """
Question 2:
Now let's compute the duration variable. It should contain the duration of a ride in minutes.
What's the standard deviation of the trips duration in January?"""
    print(q2)
    yellow_taxi_data['datetime_difference'] = yellow_taxi_data.tpep_dropoff_datetime - yellow_taxi_data.tpep_pickup_datetime
    yellow_taxi_data['duration'] = yellow_taxi_data['datetime_difference'].astype('int64') / ( 60 * 10 ** 6)
    print(yellow_taxi_data['duration'].head())
    std = yellow_taxi_data['duration'].std()
    print(f"The standard deviation of the taxi trip duration (in minutes) is: {std}")

    q3 = """
Question 3:
Next, we need to check the distribution of the duration variable. There are some outliers. 
Let's remove them and keep only the records where the duration was between 1 and 60 minutes (inclusive).
What fraction of the records are left after you dropped the outliers?"""
    print(q3)
    total_rows = yellow_taxi_data.shape[0]
    yellow_taxi_data = yellow_taxi_data[(yellow_taxi_data['duration'] <= 60) & (yellow_taxi_data['duration'] >= 1)]
    total_rows_after_dropping_outliers = yellow_taxi_data.shape[0]
    print(f"Cols before: {total_rows}, cols after: {total_rows_after_dropping_outliers}")
    print(f"Percent remaining: {100 * total_rows_after_dropping_outliers/total_rows:.2f}%")

    q4 = """
Question 4:
Let's apply one-hot encoding to the pickup and dropoff location IDs. We'll use only these two features for our model.

Turn the dataframe into a list of dictionaries (remember to re-cast the ids to strings - otherwise it will label encode them)
Fit a dictionary vectorizer
Get a feature matrix from it
What's the dimensionality of this matrix (number of columns)?"""

    print(q4)
    categorical = ['PULocationID', 'DOLocationID']

    print(f"Casting categorical variables {categorical} as strings...")
    yellow_taxi_data[categorical] = yellow_taxi_data[categorical].astype(str)

    print(f"Creating train dicts. for {[categorical]}..")
    train_dicts = yellow_taxi_data[categorical].to_dict(orient='records')

    x_train = dv.fit_transform(train_dicts)
    print(f"{x_train.shape=}")

    q5 = """
Now let's use the feature matrix from the previous step to train a model.

Train a plain linear regression model with default parameters, where duration is the response variable
Calculate the RMSE of the model on the training data
What's the RMSE on train?
"""

    target = 'duration'
    y_train = yellow_taxi_data[target].values

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_train)
    rmse = root_mean_squared_error(y_train, y_pred)
    print(f"RMSE for LR model: {rmse}")

    q6 = """
Now let's apply this model to the validation dataset (February 2023).

What's the RMSE on validation?
"""
    validation_data = pd.read_parquet("01-intro/homework/resources/yellow_tripdata_2023-02.parquet")
    validation_data = drop_outliers(validation_data)
    x_val = process_data(validation_data, categorical)
    y_val_pred = lr.predict(x_val)
    y_val_true = validation_data[target].values
    rmse_val = root_mean_squared_error(y_val_true, y_val_pred)
    print(f"RMSE of validation data: {rmse_val}")


