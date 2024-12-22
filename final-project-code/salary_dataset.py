import pandas as pd
import os
import plotly.graph_objects as go
import warnings
from utils import save_to_text_file

LOGS_FOLDER = "logs"
GRAPH_FOLDER = "graphs"


# Remove duplicates
def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the DataFrame.
    """
    return data.drop_duplicates()


# Convert columns to categorical
def convert_to_categorical(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Converts the specified columns to categorical data type.
    """
    for col in columns:
        data.loc[:, col] = data[col].astype("category")
    return data


# Preprocess the data
def pre_process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies data preprocessing steps including removing duplicates and converting columns to categorical.
    """
    data_no_duplicate = remove_duplicates(data)
    data_cleaned = convert_to_categorical(data_no_duplicate, CATEGORICAL_COLUMNS)
    return data_cleaned


# Calculate statistics
def calculate_statistics(data: pd.DataFrame, numerical_columns: list) -> pd.DataFrame:
    """
    Calculates and returns a summary of statistics including Range, Mean, Mode, Standard Deviation, and IQR for numerical columns.
    """
    summary_df = pd.DataFrame(
        columns=numerical_columns,
        index=["Range", "Mean", "Mode", "Standard Deviation", "IQR"],
    )
    for col in numerical_columns:
        column_range = [float(data[col].min()), float(data[col].max())]
        column_mean = float(data[col].mean())
        column_mode = data[col].mode()[0] if not data[col].mode().empty else None
        column_std = float(data[col].std())
        column_iqr = float(data[col].quantile(0.75) - data[col].quantile(0.25))
        summary_df.loc["Range", col] = str(column_range)
        summary_df.loc["Mean", col] = column_mean
        summary_df.loc["Mode", col] = column_mode
        summary_df.loc["Standard Deviation", col] = column_std
        summary_df.loc["IQR", col] = column_iqr
    return summary_df


# Map experience levels
def map_experience_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps experience level codes to more descriptive labels.
    """
    experience_mapping = {
        "EN": "Entry-level",
        "MI": "Mid-level",
        "SE": "Senior-level",
        "EX": "Executive-level",
    }
    df.loc[:, "experience_level"] = df["experience_level"].map(experience_mapping)
    return df


# Generate Graph 1: Salary Distribution Plot
def generate_salary_distribution_plot(data: pd.DataFrame) -> None:
    """
    Generates and saves a histogram of salary distribution in USD.
    """
    salary_histogram = go.Histogram(
        x=data["salary_in_usd"], nbinsx=50, marker_color="blue", opacity=0.75
    )
    fig = go.Figure(data=[salary_histogram])
    fig.update_layout(
        title="Salary Distribution (in USD)",
        xaxis_title="Salary in USD",
        yaxis_title="Count",
        bargap=0.2,
    )
    save_graph(fig, "T0_1_salary_distribution_plot_1.png", "Histogram")


# Generate Graph 2: Salary Distribution by Experience Level Boxplot
def generate_salary_experience_boxplot(data: pd.DataFrame) -> None:
    """
    Generates and saves a boxplot showing salary distribution by experience level.
    """
    salary_experience_boxplot = go.Box(
        x=data["experience_level"],
        y=data["salary_in_usd"],
        boxmean="sd",
        marker_color="green",
    )
    fig = go.Figure(data=[salary_experience_boxplot])
    fig.update_layout(
        title="Salary Distribution by Experience Level",
        xaxis_title="Experience Level",
        yaxis_title="Salary in USD",
    )
    save_graph(fig, "T0_2_salary_experience_boxplot_2.png", "Box Plot")


# Generate Graph 3: Top 10 Job Titles by Salary
def generate_top_job_titles_salary_plot(data: pd.DataFrame) -> None:
    """
    Generates and saves a boxplot of the top 10 job titles by average salary in USD.
    """
    top_10_job_titles = (
        data.groupby("job_title")["salary_in_usd"].mean().nlargest(10).index
    )
    filtered_data = data[data["job_title"].isin(top_10_job_titles)]
    top_job_titles_boxplot = go.Box(
        x=filtered_data["job_title"],
        y=filtered_data["salary_in_usd"],
        boxmean="sd",
        marker_color="purple",
    )
    fig = go.Figure(data=[top_job_titles_boxplot])
    fig.update_layout(
        title="Top 10 Job Titles by Salary",
        xaxis_title="Job Title",
        yaxis_title="Salary in USD",
    )
    save_graph(fig, "T0_3_top_job_titles_salary_plot_3.png", "Box Plot")


# Generate Graph 4: Salary Trend Line Plot
def generate_salary_trend_lineplot(data: pd.DataFrame) -> None:
    """
    Generates and saves a line plot showing the trend of average salary over the years.
    """
    yearly_salary_avg = data.groupby("work_year")["salary_in_usd"].mean().reset_index()
    salary_trend_lineplot = go.Scatter(
        x=yearly_salary_avg["work_year"],
        y=yearly_salary_avg["salary_in_usd"],
        mode="lines+markers",
        marker=dict(size=10, color="blue"),
        line=dict(color="green"),
    )
    fig = go.Figure(data=[salary_trend_lineplot])
    fig.update_layout(
        title="Salary Trend Over the Years",
        xaxis=dict(title="Year", type="linear", dtick=1, tickformat="d"),
        yaxis=dict(title="Average Salary in USD", dtick=10000, tickformat=",d"),
    )
    save_graph(fig, "T0_4_salary_trend_lineplot_integer_4.png", "Line Plot")


# Generate Graph 5: Job Count Barplot by Salary Bracket
def generate_job_count_barplot(data: pd.DataFrame) -> None:
    """
    Generates and saves a stacked bar chart showing the count of jobs by salary bracket over the years.
    """
    salary_bins = [0, 50000, 100000, 150000, 200000, 250000, 300000]
    salary_labels = [
        "0-50k",
        "50k-100k",
        "100k-150k",
        "150k-200k",
        "200k-250k",
        "250k+",
    ]
    data.loc[:, "salary_bracket"] = pd.cut(
        data["salary_in_usd"],
        bins=salary_bins,
        labels=salary_labels,
        include_lowest=True,
    )
    job_count_by_year_salary = (
        data.groupby(["work_year", "salary_bracket"])
        .size()
        .reset_index(name="job_count")
    )
    job_count_bar = [
        go.Bar(
            x=job_count_by_year_salary[
                job_count_by_year_salary["salary_bracket"] == label
            ]["work_year"],
            y=job_count_by_year_salary[
                job_count_by_year_salary["salary_bracket"] == label
            ]["job_count"],
            name=label,
        )
        for label in salary_labels
    ]
    fig = go.Figure(data=job_count_bar)
    fig.update_layout(
        barmode="stack",
        title="Job Count Increase in Data Science Roles by Salary Bracket",
        xaxis_title="Year",
        yaxis_title="Job Count",
        hovermode="closest",
    )
    save_graph(fig, "T0_5_job_count_salary_bracket_barplot_5.png", "Stacked Bar Chart")


# Save graph as PNG
def save_graph(fig: go.Figure, file_name: str, graph_type: str) -> None:
    """
    Saves the given Plotly graph as a PNG file.
    """
    root_folder = os.getcwd()
    save_path = os.path.join(root_folder, GRAPH_FOLDER, file_name)
    fig.write_image(save_path)


# Ignore all warnings
warnings.filterwarnings("ignore")

CATEGORICAL_COLUMNS = [
    "experience_level",
    "employment_type",
    "job_title",
    "salary_currency",
    "employee_residence",
    "company_location",
    "company_size",
]
NUMERICAL_COLUMNS = ["salary", "salary_in_usd", "remote_ratio", "work_year"]


def data_science_salary_dataset(df):

    # Preprocess the data
    cleaned_data = pre_process_data(df)
    cleaned_data = map_experience_level(cleaned_data)

    # Calculate and print statistics
    statistics = calculate_statistics(cleaned_data, NUMERICAL_COLUMNS)
    save_to_text_file(statistics.to_string(), LOGS_FOLDER, "T0_salary_statistics.txt")

    # Generate various plots
    generate_salary_distribution_plot(cleaned_data)
    generate_salary_experience_boxplot(cleaned_data)
    generate_top_job_titles_salary_plot(cleaned_data)
    generate_salary_trend_lineplot(cleaned_data)
    generate_job_count_barplot(cleaned_data)
