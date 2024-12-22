import os
import numpy as np


def save_to_text_file(content, folder="logs", filename="output.txt"):
    """
    Save any content to a text file.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Define the file path
    file_path = os.path.join(folder, filename)

    # Write the content to the file
    with open(file_path, "w") as file:
        file.write(content)

    return file_path


def save_plot(plt, folder="graphs", filename="plot.png"):
    """
    Save a matplotlib plot to a specified folder with a given filename.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Full path to the file
    file_path = os.path.join(folder, filename)

    # Save the plot
    plt.savefig(file_path)

    # Optionally close the plot to free memory
    plt.close()


def convert_nparray_to_string(arr):
    return np.array2string(arr)


def calculate_and_save_metrics_reg(metrics, folder="logs", filename="metrics.txt"):
    """
    Calculate evaluation metrics (MAE, MAPE, MASE, MSE, RMSE, R²) and save them to a text file.
    """
    # Prepare content for the text file
    content = "\n".join([f"{key}: {value:.5f}" for key, value in metrics.items()])

    # Save content to a text file
    save_to_text_file(content, folder=folder, filename=filename)


def calculate_and_save_metrics(y_true, y_pred, folder="logs", filename="metrics.txt"):
    """
    Calculate evaluation metrics (Accuracy, Precision, Recall, F1 Score, MAE) and save them to a text file.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        mean_absolute_error,
    )

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, average="weighted") * 100,
        "Recall": recall_score(y_true, y_pred, average="weighted") * 100,
        "F1 Score": f1_score(y_true, y_pred, average="weighted") * 100,
        "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
    }

    # Prepare content for the text file
    content = "\n".join([f"{key}: {value:.2f}" for key, value in metrics.items()])

    # Save content to a text file
    save_to_text_file(content, folder=folder, filename=filename)
