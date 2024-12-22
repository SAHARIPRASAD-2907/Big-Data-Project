import pandas as pd
import os
import sys
from preprocess_data import preprocess_data, correlation_analysis
from timeseries_analysis import time_series_data_analysis
from regression_analysis import regression_data_analysis
from anomaly_detection import anomaly_data_analysis
from cluster_analysis import cluster_data_analysis
from salary_dataset import data_science_salary_dataset


# TASK0: OLD Dataset
def salary_dataset_analysis(path):
    # Read Dataset
    data = pd.read_csv(path, low_memory=False)
    # Process dataset
    data_science_salary_dataset(data)


def pre_processing_analysis(path):
    # Read Dataset
    data = pd.read_csv(path, low_memory=False)
    # Preprocess data
    data = preprocess_data(data)
    # Corelation Analysis
    data = correlation_analysis(data)
    # Save Pre-processed data
    data.to_csv("pre_processed_data.csv")


# Task2: Time Series Analysis
def time_series_analysis(path):
    # Read Pre-processed Dataset
    data = pd.read_csv(path, low_memory=False, parse_dates=["time"])
    data.set_index("time", inplace=True)
    data = data.drop(columns=["year_month"], axis=1)
    # Perform Time series analysis
    time_series_data_analysis(data)


# Task3: Regression Analysis
def regression_analysis(path):
    # Read Pre-processed Dataset
    data = pd.read_csv(path, low_memory=False, parse_dates=["time"])
    data.set_index("time", inplace=True)
    # Perform Regression analysis
    regression_data_analysis(data)


# Task4: Anomaly Detection
def anomaly_detection(path):
    # Read Pre-processed Dataset
    data = pd.read_csv(path, low_memory=False, parse_dates=["time"])
    data.set_index("time", inplace=True)
    data = data.drop(columns=["year_month"], axis=1)
    # Perform Regression analysis
    anomaly_data_analysis(data)


# Task5: Cluster Analysis
def cluster_analysis(path):
    # Read Dataset
    data = pd.read_csv(path, low_memory=False)
    # Cluster Analysis
    cluster_data_analysis(data)


if __name__ == "__main__":
    # Check if file path is provided
    if len(sys.argv) != 4:
        print(
            "Usage: python3 report_2_task.py <file_path_1> <file_path_2> <file_path_3>"
        )
        print(
            "path_1 = ds_salaries.csv, path_2 = HomeC.csv, path_3 = sample_home_data.csv"
        )
        sys.exit(1)

    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]
    file_path_3 = sys.argv[3]
    # Create Logs and graphs folder
    os.makedirs("logs", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)
    # Task0: Analysis of previous dataset
    print("TASK0: Analysis of Salary Dataset")
    salary_dataset_analysis(file_path_1)
    # TASK1: Pre-processing and Co-Relation Analysis
    print("TASK1: Pre processing of Smart Home Dataset")
    pre_processing_analysis(file_path_2)
    # TASK2: Time Series Analysis
    print("TASK2: Time Series Data Analysis")
    time_series_analysis(file_path_3)
    # TASK3: Regression Analysis
    print("TASK3: Regression Analysis")
    regression_analysis(file_path_3)
    # Task4: Anomaly Detection
    print("TASK4: Anomaly Detection")
    anomaly_detection(file_path_3)
    # Task5: Cluster Analysis
    print("TASK5: Cluster Analysis")
    cluster_analysis(file_path_2)
