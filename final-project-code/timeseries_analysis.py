import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from utils import save_plot

warnings.filterwarnings("ignore")


LOGS_FOLDER = "logs"
GRAPH_FOLDER = "graphs"


# First 6 Energy Columns
def energy_plot_1(data: pd.DataFrame):
    """
    The following code was written by referring the following
    https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/
    """
    # Resample data and compute the daily mean for the first 6 columns
    resampled_data = data[data.columns[:6]].resample("D").mean()

    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 12), constrained_layout=True)
    axes = axes.flatten()

    # Generate a colormap for dark colors
    dark_colors = cm.get_cmap("Dark2_r", 6)

    # Plot each column in its own subplot
    for i, column in enumerate(resampled_data.columns):
        color = dark_colors(i)
        axes[i].plot(
            resampled_data.index, resampled_data[column], label=column, color=color
        )
        axes[i].set_title(column)
        axes[i].grid(True)
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_ylabel("Energy (kWh)")
        axes[i].legend()
        axes[i].xaxis.set_major_locator(mdates.MonthLocator())
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    for j in range(len(resampled_data.columns[:13]), len(axes)):
        plt.delaxes(axes[j])

    plt.suptitle("Energy Time Series Analysis-1")
    save_plot(plt, GRAPH_FOLDER, "T2_1_energy_time_series_first.png")


# First 6 Energy Columns
def energy_plot_2(data: pd.DataFrame):
    """
    The following code was written by referring the following
    https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/
    """
    # Resample data and compute the daily mean for the first 6 columns
    resampled_data = data[data.columns[6:13]].resample("D").mean()

    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(22, 12), constrained_layout=True
    )
    axes = axes.flatten()

    # Generate a colormap for dark colors
    dark_colors = cm.get_cmap("Dark2_r", 7)

    # Plot each column in its own subplot
    for i, column in enumerate(resampled_data.columns):
        color = dark_colors(i)
        axes[i].plot(
            resampled_data.index, resampled_data[column], label=column, color=color
        )
        axes[i].set_title(column)
        axes[i].grid(True)
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_ylabel("Energy (kWh)")
        axes[i].legend()
        axes[i].xaxis.set_major_locator(mdates.MonthLocator())
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    # Delete 7th
    fig.delaxes(axes[7])
    # Delete 8th
    fig.delaxes(axes[8])
    plt.suptitle("Energy Time Series Analysis-2")
    save_plot(plt, GRAPH_FOLDER, "T2_2_energy_time_series_second.png")


# Whether plot
def weather_plots(data):
    """
    The following code was written by referring the following
    https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/
    """
    # Resample data and compute the daily mean for weather columns
    resampled_data = data[data.columns[13:]].resample("D").mean()

    fig, axes = plt.subplots(
        nrows=4, ncols=3, figsize=(22, 12), constrained_layout=True
    )
    axes = axes.flatten()

    # Generate a colormap for dark colors
    dark_colors = cm.get_cmap("Dark2_r", 12)

    # Plot each column in its own subplot
    for i, column in enumerate(resampled_data.columns):
        color = dark_colors(i)
        axes[i].plot(
            resampled_data.index, resampled_data[column], label=column, color=color
        )
        axes[i].set_title(column)
        axes[i].grid(True)
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_ylabel("Energy (kWh)")
        axes[i].legend()
        axes[i].xaxis.set_major_locator(mdates.MonthLocator())
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    # Delete 8th
    fig.delaxes(axes[11])
    plt.suptitle("Weather Time Series Analysis-1")
    save_plot(plt, GRAPH_FOLDER, "T2_3_weather_time_series.png")


def average_consumption_by_weekday(data):
    """
    The following code was written by referring the following
    https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/
    """
    """
    Plot the average consumption per day of the week for the first 13 columns.
    """
    data["weekday"] = data.index.day_name()
    # Define the order of days of the week
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Group data by weekday and calculate the mean
    mean_weekday = (
        data.groupby("weekday")
        .agg({i: "mean" for i in data.columns[:-5].tolist()})
        .reindex(days)
    )

    # Prepare the subplot layout
    _, axes = plt.subplots(
        nrows=-(
            -len(mean_weekday.columns[:13]) // 3
        ),  # Calculate rows for 3 columns per row
        ncols=3,
        figsize=(15, 10),
        constrained_layout=True,
    )
    axes = axes.flatten()

    # Generate a colormap for dark colors
    dark_colors = cm.get_cmap("Dark2_r", len(mean_weekday.columns[:13]))

    # Plot each column in its own subplot
    for i, column in enumerate(mean_weekday.columns[:13]):
        color = dark_colors(i)
        axes[i].plot(
            mean_weekday.index,
            mean_weekday[column],
            label=column,
            color=color,
            marker="o",
        )
        axes[i].set_title(column)
        axes[i].grid(True)
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].legend()

    # Hide unused subplots
    for j in range(len(mean_weekday.columns[:13]), len(axes)):
        plt.delaxes(axes[j])

    # Add a common title
    plt.suptitle("Average Consumption per Day of the Week (Energy in Kwh)", fontsize=16)

    # Save the plot
    save_plot(plt, GRAPH_FOLDER, "T2_4_energy_consumption_weekly.png")


def average_consumption_by_hour(data):
    """
    The following code was written by referring the following
    https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-to-time-series-analysis/
    """
    data["hour"] = data.index.hour
    # Group data by hour and calculate the mean
    mean_hour = data.groupby("hour").agg(
        {i: "mean" for i in data.columns[:-5].tolist()}
    )

    # Prepare the subplot layout
    _, axes = plt.subplots(
        nrows=-(
            -len(mean_hour.columns[:13]) // 3
        ),  # Calculate rows for 3 columns per row
        ncols=3,
        figsize=(18, 12),
        constrained_layout=True,
    )
    axes = axes.flatten()

    # Generate a colormap for dark colors
    dark_colors = cm.get_cmap("Dark2_r", len(mean_hour.columns[:13]))

    # Plot each column in its own subplot
    for i, column in enumerate(mean_hour.columns[:13]):
        color = dark_colors(i)
        axes[i].plot(
            mean_hour.index, mean_hour[column], label=column, color=color, marker="o"
        )
        axes[i].set_title(column)
        axes[i].grid(True)
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].legend()

    # Hide unused subplots
    for j in range(len(mean_hour.columns[:13]), len(axes)):
        plt.delaxes(axes[j])

    # Add a common title
    plt.suptitle("Average Consumption per Hour (Energy in Kwh)", fontsize=16)

    # Save the plot
    save_plot(plt, GRAPH_FOLDER, "T2_5_energy_consumption_hourly.png")


def time_series_data_analysis(data: pd.DataFrame):
    # Plotting Energy plot for first 6
    energy_plot_1(data)
    # Plotting Energy plot for next 6
    energy_plot_2(data)
    # Plot Weather plots
    weather_plots(data)
    # Plot Consumption per week data
    average_consumption_by_weekday(data)
    # Plot hourly consumption
    average_consumption_by_hour(data)
