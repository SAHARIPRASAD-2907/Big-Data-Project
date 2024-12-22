import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import save_plot, calculate_and_save_metrics_reg
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


warnings.filterwarnings("ignore")

LOGS_FOLDER = "logs"
GRAPH_FOLDER = "graphs"


def baseline_model_regression(data):
    """
    The following code was written by referring the following link
    https://medium.com/wearesinch/simple-anomaly-detection-algorithms-for-streaming-data-machine-learning-92cfaeb6f43b
    """
    data_daily = data["House overall"].resample("d").mean().fillna(0)
    # Define la moving average
    baseline = data_daily.rolling(window=10).mean()
    # Split Training and testing data
    size = int(len(data_daily) * 0.7)
    train = data_daily[:size]
    test = data_daily[size:]
    # Using Moving Average
    baseline_test = baseline.loc[test.index[0] :]
    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(train, c="green", label="Train Data")
    plt.plot(test, c="blue", label="Test Data")
    plt.plot(baseline_test, c="red", label="Rolling Mean")
    plt.legend(fontsize=12)
    plt.ylabel("Energy kW")
    plt.xlabel("Months")
    plt.title("Train and Test Data with Rolling Mean")
    plt.grid()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.xticks(rotation=45)
    plt.tight_layout()  # Prevents axis cropping
    # Save plot
    save_plot(plt, GRAPH_FOLDER, "T3_1_baseline_model_regression.png")
    # Save Metrics
    metrics = {
        "MAE": mean_absolute_error(test, baseline_test),
        "MAPE": np.mean(np.abs(baseline_test - test) / np.abs(test)) * 100,
        "MSE": mean_squared_error(test, baseline_test),
        "RMSE": np.sqrt(mean_squared_error(test, baseline_test)),
        "R²": r2_score(test, baseline_test),
    }
    calculate_and_save_metrics_reg(
        metrics, LOGS_FOLDER, "T3_1_metrics_baseline_model.txt"
    )


def airma_model_regression(data):
    """
    The following code was written by referring the following link
    https://medium.com/aimonks/anomaly-detection-for-time-series-analysis-eeecd6282f53
    """
    data_daily = data["House overall"].resample("d").mean().fillna(0)
    X = data_daily.values
    size = int(len(X) * 0.7)
    train, test = X[:size], X[size:]

    model = ARIMA(train, order=(2, 0, 1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test)).tolist()
    conf_int = model_fit.get_forecast(steps=len(test)).conf_int(0.05)

    index_test = data_daily[size:].index
    predictions_series = pd.Series(forecast, index=index_test)
    confidence = pd.DataFrame(conf_int, columns=["lower", "upper"], index=index_test)

    plt.figure(figsize=(15, 4))
    plt.plot(data_daily[:size], c="green", label="Train Data")
    plt.plot(data_daily[size:], c="blue", label="Test Data")
    plt.plot(predictions_series, c="red", label="Predictions")
    plt.fill_between(
        confidence.index,
        confidence["lower"],
        confidence["upper"],
        color="k",
        alpha=0.15,
    )
    plt.legend()
    plt.grid()
    plt.margins(x=0)
    plt.title("ARIMA Model Results on Test Data")
    plt.ylabel("Energy kW")
    plt.xlabel("Months")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.xticks(rotation=45)
    save_plot(plt, GRAPH_FOLDER, "T3_2_AIRMA_model_regression.png")
    # Save Metrics
    metrics = {
        "MAE": mean_absolute_error(test, forecast),
        "MAPE": np.mean(np.abs(forecast - test) / np.abs(test)) * 100,
        "MSE": mean_squared_error(test, forecast),
        "RMSE": np.sqrt(mean_squared_error(test, forecast)),
        "R²": r2_score(test, forecast),
    }
    calculate_and_save_metrics_reg(
        metrics, LOGS_FOLDER, "T3_2_metrics_baseline_model.txt"
    )


def sarmax_model_regression(data):
    """
    The following code was written by referring the following link
    https://victoriametrics.com/blog/victoriametrics-anomaly-detection-handbook-chapter-3/
    """
    data["month"] = data.index.month
    data["weekday"] = data.index.day_name()
    data_exog = pd.get_dummies(
        data, columns=["month", "weekday"], prefix=["month", "weekday"]
    )
    ext_var_list = [
        "month_1",
        "month_2",
        "month_3",
        "month_4",
        "month_5",
        "month_6",
        "month_7",
        "month_8",
        "month_9",
        "month_10",
        "month_11",
        "month_12",
        "weekday_Friday",
        "weekday_Monday",
        "weekday_Saturday",
        "weekday_Sunday",
        "weekday_Thursday",
        "weekday_Tuesday",
        "weekday_Wednesday",
    ]
    exog_part = data_exog[ext_var_list]
    exog_part = exog_part.resample("d").mean().fillna(0)
    # Train and Test Data
    data_daily = data["House overall"].resample("d").mean().fillna(0)
    size = int(len(data_daily) * 0.7)
    train = data_daily[:size]
    test = data_daily[size:]

    model = sm.tsa.statespace.SARIMAX(
        endog=train,
        exog=exog_part[:size],
        order=(2, 1, 1),
        seasonal_order=(5, 0, 1, 12),
    )
    model_fit = model.fit(disp=0)
    forecast = model_fit.forecast(steps=len(test), exog=exog_part[size:]).tolist()
    conf_int = model_fit.get_forecast(steps=len(test), exog=exog_part[size:]).conf_int(
        0.05
    )

    index_test = data_daily[size:].index
    predictions_series = pd.Series(forecast, index=index_test)
    confidence = pd.DataFrame(conf_int, columns=["lower", "upper"], index=index_test)

    plt.figure(figsize=(15, 5))
    plt.plot(data_daily[:size], c="green", label="Train Data")
    plt.plot(data_daily[size:], c="blue", label="Test Data")
    plt.plot(predictions_series, c="red", label="Predictions")
    plt.ylabel("Energy kW")
    plt.xlabel("Months")
    plt.fill_between(
        confidence.index,
        confidence["lower"],
        confidence["upper"],
        color="k",
        alpha=0.15,
    )
    plt.legend()
    plt.grid()
    plt.margins(x=0)
    plt.title("SARMAX Model Results on Test Data")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.xticks(rotation=45)
    save_plot(plt, GRAPH_FOLDER, "T3_3_SARMAX_model_regression.png")
    # Save Metrics
    metrics = {
        "MAE": mean_absolute_error(test, forecast),
        "MAPE": np.mean(np.abs(forecast - test) / np.abs(test)) * 100,
        "MSE": mean_squared_error(test, forecast),
        "RMSE": np.sqrt(mean_squared_error(test, forecast)),
        "R²": r2_score(test, forecast),
    }
    calculate_and_save_metrics_reg(
        metrics, LOGS_FOLDER, "T3_3_metrics_sarmax_model.txt"
    )


def get_train_test_data_for_lstm(data_daily):
    # Normalize Features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_daily[data_daily.columns[1:]] = scaler.fit_transform(
        data_daily[data_daily.columns[1:]]
    )
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    data_daily[["House overall"]] = scaler_target.fit_transform(
        data_daily[["House overall"]]
    )
    # Split data into training and testing
    size = int(len(data_daily) * 0.7)
    data_daily_train = data_daily[:size]
    data_daily_test = data_daily[size:]
    X_train, X_test = [], []
    Y_train, Y_test = [], []
    n_past = 1
    n_future = 1
    for i in range(n_past, len(data_daily_train) - n_future + 1):
        X_train.append(data_daily_train.iloc[i - n_past : i, 0 : data_daily.shape[1]])
        Y_train.append(data_daily_train.iloc[i + n_future - 1 : i + n_future, 0])
    for i in range(n_past, len(data_daily_test) - n_future + 1):
        X_test.append(
            data_daily_test.iloc[i - n_past : i, 0 : data_daily_test.shape[1]]
        )
        Y_test.append(data_daily_test.iloc[i + n_future - 1 : i + n_future, 0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    return n_past, scaler_target, X_train, X_test, Y_train, Y_test


def lstm_n_layers(data, num_layers, plot_file_name, metric_file_name):
    """
    The following code was written by referring the following link
    https://medium.com/@zhonghong9998/anomaly-detection-in-time-series-data-using-lstm-autoencoders-51fd14946fa3
    """
    data_daily = data[
        [
            "House overall",
            "Furnace",
            "Living room",
            "Barn",
            "temperature",
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
    data_daily = data_daily.resample("D").mean().fillna(0)
    # Get Train and test
    n_past, scaler_target, X_train, X_test, Y_train, Y_test = (
        get_train_test_data_for_lstm(data_daily)
    )
    # Setup Model
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.LSTM(
            num_layers,
            activation="relu",
            return_sequences=False,
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )
    )
    model.add(tf.keras.layers.Dense(Y_train.shape[1]))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, Y_train, epochs=80, verbose=0)
    size = int(len(data_daily) * 0.7)
    # Invert scaling
    data_daily[["House overall"]] = scaler_target.inverse_transform(
        data_daily[["House overall"]]
    )
    Train_pred = model.predict(X_train, verbose=0)
    Y_pred = model.predict(X_test, verbose=0)
    Y_pred = scaler_target.inverse_transform(Y_pred)
    Train_pred = scaler_target.inverse_transform(Train_pred)
    # Test generation
    Y_pred_series = pd.Series(
        Y_pred.flatten().tolist(), index=data_daily["House overall"][size:-n_past].index
    )
    Train_pred_series = pd.Series(
        Train_pred.flatten().tolist(),
        index=data_daily["House overall"][n_past:size].index,
    )
    plt.figure(figsize=(15, 4))
    plt.plot(data_daily["House overall"][:-n_past], c="blue", label="data")
    plt.plot(Y_pred_series, c="red", label="model test")
    plt.plot(Train_pred_series, c="green", label="model train")
    plt.legend()
    plt.ylabel("Energy kW")
    plt.xlabel("Months")
    plt.title(f"LSTM {num_layers} layers Model Results on Test Data")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.grid(), plt.margins(x=0)
    save_plot(plt, GRAPH_FOLDER, plot_file_name)
    Y_test = data_daily["House overall"][size:-n_past]
    # Save Metrics
    metrics = {
        "MAE": mean_absolute_error(Y_pred, Y_test),
        "MAPE": np.mean(np.abs(Y_pred[:, 0] - Y_test.values) / np.abs(Y_test.values)),
        "MSE": mean_squared_error(Y_test, Y_pred),
        "RMSE": np.sqrt(mean_squared_error(Y_test, Y_pred)),
        "R²": r2_score(Y_test, Y_pred),
    }
    calculate_and_save_metrics_reg(metrics, LOGS_FOLDER, metric_file_name)


def regression_data_analysis(data):
    # Baseline model Moving Average
    baseline_model_regression(data)
    # AIRMA Model
    airma_model_regression(data)
    # SARIMAX with exogas
    sarmax_model_regression(data)
    # LSTM With 10 Layers
    lstm_n_layers(
        data,
        10,
        "T3_4_LSTM_10_model_regression.png",
        "T3_4_LSTM_10_model_regression.txt",
    )
    # LSTM With 20 Layers
    lstm_n_layers(
        data,
        20,
        "T3_5_LSTM_20_model_regression.png",
        "T3_5_LSTM_20_model_regression.txt",
    )
    # LSTM with 30 Layers
    lstm_n_layers(
        data,
        30,
        "T3_6_LSTM_30_model_regression.png",
        "T3_6_LSTM_30_model_regression.txt",
    )
