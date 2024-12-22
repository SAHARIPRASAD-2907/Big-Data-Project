import pandas as pd
from utils import save_to_text_file, save_plot, convert_nparray_to_string
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


LOGS_FOLDER = "logs"
GRAPH_FOLDER = "graphs"


def convert_unix_time_stamp(data: pd.DataFrame):
    """
    Convert Unix to Time stamp data
    """
    data["time"] = pd.to_datetime(data["time"], unit="s")
    data["time"] = pd.DatetimeIndex(
        pd.date_range("2016-01-01 05:00", periods=len(data), freq="min")
    )
    data = data.set_index("time")
    return data


def preprocess_data(data: pd.DataFrame):
    # Delete Last Row as it has Nan Value
    data = data[:-1]
    # Save data info
    buffer = StringIO()
    data.info(buf=buffer)
    save_to_text_file(buffer.getvalue(), LOGS_FOLDER, "T1_1_data_info.txt")
    # View First 5 rows
    save_to_text_file(data.head().to_string(), LOGS_FOLDER, "T1_2_data_top_rows.txt")
    # Convert to unix time stamp
    data = convert_unix_time_stamp(data)
    # Get Unique Summary values
    save_to_text_file(
        convert_nparray_to_string(data["summary"].unique()),
        LOGS_FOLDER,
        "T1_3_unique_summary.txt",
    )
    # Get Unique Cloud cover values
    save_to_text_file(
        convert_nparray_to_string(data["cloudCover"].unique()),
        LOGS_FOLDER,
        "T1_4_unique_cloudCover.txt",
    )
    # Update '[kW]' in columns name, sum similar consumptions and delete 'summary' column
    data.columns = [i.replace(" [kW]", "") for i in data.columns]
    # Group Furnace and Kitchen columns
    data["Furnace"] = data[["Furnace 1", "Furnace 2"]].sum(axis=1)
    data["Kitchen"] = data[["Kitchen 12", "Kitchen 14", "Kitchen 38"]].sum(
        axis=1
    )  # We could also use the mean
    # Drop After Grouping
    data.drop(
        [
            "Furnace 1",
            "Furnace 2",
            "Kitchen 12",
            "Kitchen 14",
            "Kitchen 38",
            "icon",
            "summary",
        ],
        axis=1,
        inplace=True,
    )
    # Replace Invalid values 'cloudCover' with backfill method
    data["cloudCover"].replace(["cloudCover"], method="bfill", inplace=True)
    data["cloudCover"] = data["cloudCover"].astype("float")
    # Reorder columns
    data = data[
        [
            "use",
            "gen",
            "House overall",
            "Dishwasher",
            "Home office",
            "Fridge",
            "Wine cellar",
            "Garage door",
            "Barn",
            "Well",
            "Microwave",
            "Living room",
            "Furnace",
            "Kitchen",
            "Solar",
            "temperature",
            "humidity",
            "visibility",
            "apparentTemperature",
            "pressure",
            "windSpeed",
            "cloudCover",
            "windBearing",
            "precipIntensity",
            "dewPoint",
            "precipProbability",
        ]
    ]
    # View rows after preprocessing
    save_to_text_file(
        data.head().to_string(), LOGS_FOLDER, "T1_5_preprocess_reorder.txt"
    )
    return data


def correlation_analysis(data):
    # Energy Correlation Analysis
    plt.subplots(figsize=(10, 8))
    sns.heatmap(
        data[data.columns[0:15].tolist()].corr(),
        annot=True,
        fmt=".2f",
        vmin=-1.0,
        vmax=1.0,
        center=0,
    )
    plt.title("Energy Correlation", fontsize=12)

    # Save fig
    save_plot(plt, GRAPH_FOLDER, "T1_1_energy_corelation.png")

    # Checking 2 columns with same data
    save_to_text_file(
        f"Check Gen and Solar columns are same: ${data['gen'].equals(data['Solar'])}",
        LOGS_FOLDER,
        "T1_6_check_gen_solar.txt",
    )
    save_to_text_file(
        f"Check use and House Overall columns are same ${data['use'].equals(data['House overall'])}",
        LOGS_FOLDER,
        "T1_7_check_use_house_overall.txt",
    )

    # Plot and show overlapping
    _, axes = plt.subplots(1, 2, figsize=(16, 8))
    data[["use", "House overall"]].resample("D").mean().plot(ax=axes[0])
    data[["gen", "Solar"]].resample("D").mean().plot(ax=axes[1])
    axes[0].title.set_text("Overlapping of values (use vs House overall)")
    axes[1].title.set_text("Overlapping of values (gen vs Solar)")
    save_plot(plt, GRAPH_FOLDER, "T1_2_Overlap_values.png")
    # Drop Use and Gen
    data.drop(["use", "gen"], axis=1, inplace=True)
    # Checking corelation for weather data
    plt.subplots(figsize=(10, 8))
    sns.heatmap(
        data[data.columns[13:].tolist()].corr(),
        annot=True,
        vmin=-1.0,
        vmax=1.0,
        center=0,
    )
    plt.title("Weather corelation Analysis", fontsize=14)
    save_plot(plt, GRAPH_FOLDER, "T1_3_weather_corelation.png")
    # Investigating weather features (temperature,apparentTemperature, dewPoint)
    _, axes = plt.subplots(1, 2, figsize=(20, 8))
    data[["temperature", "apparentTemperature", "dewPoint"]].resample("D").mean().plot(
        ax=axes[0], grid=True
    )
    data[["humidity"]].resample("D").mean().plot(ax=axes[1], grid=True)
    axes[0].title.set_text("Temperature related data")
    axes[1].title.set_text("Humidity related data")
    save_plot(plt, GRAPH_FOLDER, "T1_4_temperature_humidity_relation.png")
    # Correlation between Temp diff(apparentTemperature, temperature) and other whether columns
    data["Tdiff"] = data["apparentTemperature"] - data["temperature"]
    climate_params = data.columns[13:-1].tolist()
    list_corr = []
    for i in climate_params:
        cor = data[i].corr(data["Tdiff"])
        list_corr.append(cor)
    data_corr = pd.DataFrame(
        list(zip(climate_params, list_corr)), columns=["weather", "Tdiff_corr"]
    ).set_index("weather")
    save_to_text_file(
        data_corr.to_string(), LOGS_FOLDER, "T1_8_corr_temp_weather_data.txt"
    )
    # Graph (Tdiff and windSpeed) and (apparentTemperature vs Temperature )
    _, axes = plt.subplots(2, 1, figsize=(10, 5))
    data[["Tdiff", "windSpeed"]].resample("D").mean().plot(ax=axes[0], grid=True)
    data[["apparentTemperature", "temperature"]].resample("D").mean().plot(
        ax=axes[1], grid=True
    )
    plt.suptitle(
        "(Windspeed vs Tdiff) and (apparentTemperature vs Temperature)", fontsize=14
    )
    save_plot(plt, GRAPH_FOLDER, "T1_5_temperature_windSpeed_tdiff_relation.png")
    # Drop Tdiff
    data.drop("Tdiff", axis=1, inplace=True)
    return data
