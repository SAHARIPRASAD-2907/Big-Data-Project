from sklearn.metrics import (
    mean_absolute_error,
)
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import save_plot

warnings.filterwarnings("ignore")

LOGS_FOLDER = "logs"
GRAPH_FOLDER = "graphs"


def moving_average_anomaly(data):
    """
    The following code was written by referring the following link
    https://medium.com/wearesinch/simple-anomaly-detection-algorithms-for-streaming-data-machine-learning-92cfaeb6f43b
    """
    scale = 2
    window = 20
    data_anomaly = data.resample("d").mean()
    data_anomaly = data_anomaly[["House overall"]].fillna(0)
    rolling_mean = data_anomaly.rolling(window=window).mean()
    # Plot Figure
    plt.figure(figsize=(15, 5))
    plt.title("Moving average with window size = {}".format(window))
    plt.ylabel("Energy kW")
    plt.xlabel("Months")
    plt.plot(rolling_mean, "r", label="Rolling mean trend")
    # Plot confidence intervals for smoothed values
    mae = mean_absolute_error(data_anomaly[window:], rolling_mean[window:])
    deviation = np.std(data_anomaly[window:] - rolling_mean[window:])
    lower_bond = rolling_mean - (mae + scale * deviation)
    upper_bond = rolling_mean + (mae + scale * deviation)
    plt.plot(upper_bond, "g--", label="Upper Bond / Lower Bond")
    plt.plot(lower_bond, "g--")

    # Having the intervals, find abnormal values
    anomalies = pd.DataFrame(index=data_anomaly.index, columns=data_anomaly.columns)
    anomalies[data_anomaly < lower_bond] = data_anomaly[data_anomaly < lower_bond]
    anomalies[data_anomaly > upper_bond] = data_anomaly[data_anomaly > upper_bond]
    plt.plot(anomalies, "ro", markersize=5)

    plt.plot(data_anomaly[window:], "blue", label="Actual values")
    plt.legend(loc="upper left")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.grid(True), plt.margins(x=0)
    # Save plot
    save_plot(plt, GRAPH_FOLDER, "T4_1_moving_average_anomaly.png")


def airma_anomaly(data):
    """
    The following code was written by referring the following link
    https://medium.com/aimonks/anomaly-detection-for-time-series-analysis-eeecd6282f53
    """
    data_anomaly = data.resample("d").mean()
    model = ARIMA(data_anomaly["House overall"].fillna(0), order=(2, 1, 1))
    model_fit = model.fit()

    squared_errors = model_fit.resid
    threshold = np.mean(squared_errors) + 2 * np.std(squared_errors)
    upper_bond = model_fit.predict(dynamic=False) + threshold
    lower_bond = model_fit.predict(dynamic=False) - threshold
    anomalies = data_anomaly["House overall"][
        (data_anomaly["House overall"] < lower_bond)
        | (data_anomaly["House overall"] > upper_bond)
    ]

    plt.figure(figsize=(15, 5))
    plt.plot(
        model_fit.predict(dynamic=False) + threshold,
        c="g",
        linestyle="--",
        label="Upper bound / Lower bound",
    )
    plt.ylabel("Energy kW")
    plt.xlabel("Months")
    plt.title("ARIMA anomaly detection Analysis")
    plt.plot(model_fit.predict(dynamic=False) - threshold, c="g", linestyle="--")
    plt.plot(model_fit.predict(dynamic=False), c="red", label="model")
    plt.plot(data_anomaly["House overall"], c="blue", label="Data - House overall")
    plt.plot(anomalies, "ro", markersize=5)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.legend(), plt.grid(), plt.margins(x=0)
    # Save plot
    save_plot(plt, GRAPH_FOLDER, "T4_2_ARIMA_anomaly.png")


def sarimax_anomaly(data):
    """
    The following code was written by referring the following link
    https://victoriametrics.com/blog/victoriametrics-anomaly-detection-handbook-chapter-3/
    """
    data_anomaly = data.resample("d").mean()
    model = sm.tsa.statespace.SARIMAX(
        data_anomaly["House overall"].fillna(0),
        order=(2, 1, 1),
        seasonal_order=(5, 0, 1, 12),
    )
    model_fit = model.fit(disp=0)

    squared_errors = model_fit.resid
    threshold = np.mean(squared_errors) + 2 * np.std(squared_errors)
    upper_bond = model_fit.predict(dynamic=False) + threshold
    lower_bond = model_fit.predict(dynamic=False) - threshold
    anomalies = data_anomaly["House overall"][
        (data_anomaly["House overall"] < lower_bond)
        | (data_anomaly["House overall"] > upper_bond)
    ]

    plt.figure(figsize=(15, 5))
    plt.plot(
        model_fit.predict(dynamic=False) + threshold,
        c="g",
        linestyle="--",
        label="Upper bound / Lower bound",
    )
    plt.ylabel("Energy kW")
    plt.xlabel("Months")
    plt.title("SARIMAX anomaly detection Analysis")
    plt.plot(model_fit.predict(dynamic=False) - threshold, c="g", linestyle="--")
    plt.plot(model_fit.predict(dynamic=False), c="red", label="model")
    plt.plot(data_anomaly["House overall"], c="blue", label="Data - House overall")
    plt.plot(anomalies, "ro", markersize=5)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.legend(), plt.grid(), plt.margins(x=0)
    # Save plot
    save_plot(plt, GRAPH_FOLDER, "T4_3_SARIMAX_anomaly.png")


def lstm_n_layers_anomaly(
    data,
    num_layers,
    plot_file_name,
):
    """
    The following code was written by referring the following link
    https://medium.com/@zhonghong9998/anomaly-detection-in-time-series-data-using-lstm-autoencoders-51fd14946fa3
    """
    data_anomaly = data.resample("d").mean().fillna(0)
    data_daily = data_anomaly[
        [
            "House overall",
            "temperature",
            "Furnace",
            "Living room",
            "Barn",
            "humidity",
            "apparentTemperature",
            "pressure",
            "cloudCover",
            "windBearing",
            "precipIntensity",
            "dewPoint",
            "precipProbability",
        ]
    ]
    # Normalize the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_daily[data_daily.columns[1:]] = scaler.fit_transform(
        data_daily[data_daily.columns[1:]]
    )
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    data_daily[["House overall"]] = scaler_target.fit_transform(
        data_daily[["House overall"]]
    )
    #  Sequencing the trace like in the univariate case
    X, Y = [], []
    n_past = 1
    n_future = 1
    for i in range(n_past, len(data_daily) - n_future + 1):
        X.append(data_daily.iloc[i - n_past : i, 0 : data_daily.shape[1]])
        Y.append(data_daily.iloc[i + n_future - 1 : i + n_future, 0])
    X, Y = np.array(X), np.array(Y)
    # Model
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.LSTM(
            num_layers,
            activation="relu",
            return_sequences=False,
            input_shape=(X.shape[1], X.shape[2]),
        )
    )
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(Y.shape[1]))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=30, verbose=0)
    Train_pred = model.predict(X, verbose=0)
    # Invert scaling
    data_daily[["House overall"]] = scaler_target.inverse_transform(
        data_daily[["House overall"]]
    )
    Train_pred = scaler_target.inverse_transform(Train_pred)
    Train_pred_series = pd.Series(
        Train_pred.flatten().tolist(), index=data_daily["House overall"][n_past:].index
    )
    # Calculate upper and lower bound
    mae = mean_absolute_error(data_daily["House overall"][1:], Train_pred_series)
    deviation = np.std(data_daily["House overall"][1:] - Train_pred_series)
    lower_bond = Train_pred_series - (mae + 2.5 * deviation)
    upper_bond = Train_pred_series + (mae + 2.5 * deviation)
    anomalies = data_anomaly["House overall"][1:][
        (data_anomaly["House overall"][1:] > upper_bond)
        | (data_anomaly["House overall"][1:] < lower_bond)
    ]
    # Plot Graph
    plt.figure(figsize=(15, 5))
    plt.plot(data_daily["House overall"], c="blue", label="Home Energy Usage")
    plt.plot(Train_pred_series, c="green", label="Trained model")
    plt.fill_between(lower_bond.index, lower_bond, upper_bond, color="k", alpha=0.15)
    plt.plot(anomalies, "ro", markersize=5)
    plt.legend()
    plt.ylabel("Energy kW")
    plt.xlabel("Months")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.title(f"LSTM anomaly detection with {num_layers} input layers")
    plt.xlabel("time")
    plt.ylabel("Energy Usage")
    plt.grid(), plt.margins(x=0)
    # Save Plots
    save_plot(plt, GRAPH_FOLDER, plot_file_name)


def anomaly_data_analysis(data):
    # Moving Average Anomaly Detection
    moving_average_anomaly(data)
    # ARIMA Anomaly Detection
    airma_anomaly(data)
    # SARIMAX Anomaly Detection
    sarimax_anomaly(data)
    # LSTM 10 Layers
    lstm_n_layers_anomaly(data, 10, "T4_4_LSTM_10_anomaly.png")
    # LSTM 20 Layers
    lstm_n_layers_anomaly(data, 20, "T4_5_LSTM_20_anomaly.png")
    # LSTM 30 Layers
    lstm_n_layers_anomaly(data, 30, "T4_6_LSTM_30_anomaly.png")
