import pandas as pd
import sklearn
import os
import numpy as np
from sklearn.feature_extraction import DictVectorizer

def ingest_parquet_data(filename: str) -> pd.DataFrame:
    if not os.path.exists(filename):
        raise FileNotFoundError
    return pd.read_parquet(filename)




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
    for col in yellow_taxi_data_columns:
        print(f"\t{col}")

    q2 = """
Question 2:
Now let's compute the duration variable. It should contain the duration of a ride in minutes.
What's the standard deviation of the trips duration in January?"""
    print(q2)
    yellow_taxi_data['datetime_difference'] = yellow_taxi_data.tpep_dropoff_datetime - yellow_taxi_data.tpep_pickup_datetime
    yellow_taxi_data['duration'] = yellow_taxi_data['datetime_difference'].astype('int64') / ( 60 * 10 ** 9)
    std = yellow_taxi_data['duration'].std()
    print(f"The standard deviation of the taxi trip duration (in minutes) is: {std}")

    q3 = """
Question 3:
Next, we need to check the distribution of the duration variable. There are some outliers. 
Let's remove them and keep only the records where the duration was between 1 and 60 minutes (inclusive).
What fraction of the records are left after you dropped the outliers?"""
    print(q3)
    total_cols = yellow_taxi_data.shape[0]
    yellow_taxi_data = yellow_taxi_data[(yellow_taxi_data['duration'] <= 60) & (yellow_taxi_data['duration'] >= 1)]
    total_cols_after_dropping_outliers = yellow_taxi_data.shape[0]
    print(f"Cols before: {total_cols}, cols after: {total_cols_after_dropping_outliers}")
    print(f"Percent remaining: {round(float(total_cols_after_dropping_outliers)/total_cols, 2) * 100}%")

    q4 = """
Question 4:
Let's apply one-hot encoding to the pickup and dropoff location IDs. We'll use only these two features for our model.

Turn the dataframe into a list of dictionaries (remember to re-cast the ids to strings - otherwise it will label encode them)
Fit a dictionary vectorizer
Get a feature matrix from it
What's the dimensionality of this matrix (number of columns)?"""

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    print(f"Casting categorical variables {categorical} as strings...")
    yellow_taxi_data[categorical] = yellow_taxi_data[categorical].astype(str)

    print(f"Creating train dicts. for {[categorical + numerical]}..")
    train_dicts = yellow_taxi_data[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer()
    x_train = dv.fit_transform(train_dicts)
    target = 'duration'
    y_train = yellow_taxi_data[target].values
    
    
