from sklearn.preprocessing import StandardScaler, LabelEncoder
from fcmeans import FCM
from collections import Counter
import matplotlib.pyplot as plt
from utils import save_plot, calculate_and_save_metrics
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.mixture import GaussianMixture


warnings.filterwarnings("ignore")

LOGS_FOLDER = "logs"
GRAPH_FOLDER = "graphs"


def get_data_with_pca(data):
    # Selecting a few representative numeric columns to reduce dimensionality
    selected_features = ["use [kW]", "temperature", "humidity", "pressure", "windSpeed"]
    numeric_data_subset = data[selected_features].dropna()
    # Encoding the 'summary' column
    data = data.dropna(subset=selected_features + ["summary"])
    label_encoder = LabelEncoder()
    data["summary_encoded"] = label_encoder.fit_transform(data["summary"])
    # Adding the encoded 'summary' column to the dataset
    numeric_data_subset = data[selected_features + ["summary_encoded", "summary"]]
    # Taking a smaller sample of 5,000 rows to manage memory usage
    numeric_data_sample = numeric_data_subset.sample(n=250, random_state=42)
    # Normalize the selected features (excluding the 'summary_encoded' for scaling)
    scaler = StandardScaler()
    data_normalized_sample = scaler.fit_transform(
        numeric_data_sample[selected_features]
    )
    # Apply PCA for dimensionality reduction - Adjusted to maintain 95% of the variance
    pca = PCA(n_components=0.95)
    data_pca = pca.fit_transform(data_normalized_sample)

    return data_pca, numeric_data_sample


def fuzzy_cmean_clustering(data):
    """
    The following function was written by checking the following links
    https://fuzzy-c-means.readthedocs.io/en/latest/examples/00%20-%20Basic%20clustering/
    and understanding the following link below
    https://www.kaggle.com/code/ahmaddaffaabiyyu/fuzzy-cmeans-and-hierarchical-clustering-smarthome
    """
    # Selecting a few representative numeric columns to reduce dimensionality
    selected_features = ["use [kW]", "temperature", "humidity", "pressure", "windSpeed"]
    numeric_data_subset = data[selected_features].dropna()
    # Encoding the 'summary' column
    data = data.dropna(subset=selected_features + ["summary"])
    label_encoder = LabelEncoder()
    data["summary_encoded"] = label_encoder.fit_transform(data["summary"])
    # Adding the encoded 'summary' column to the dataset
    numeric_data_subset = data[selected_features + ["summary_encoded", "summary"]]
    # Taking a smaller sample of 5,000 rows to manage memory usage
    numeric_data_sample = numeric_data_subset.sample(n=250, random_state=42)
    # Normalize the selected features (excluding the 'summary_encoded' for scaling)
    scaler = StandardScaler()
    data_normalized_sample = scaler.fit_transform(
        numeric_data_sample[selected_features]
    )
    # Fuzzy C-Means clustering
    c = 3
    fcm = FCM(n_clusters=c, m=2)
    fcm.fit(data_normalized_sample)
    # Cluster centers and labels
    labels = fcm.predict(data_normalized_sample)
    numeric_data_sample["cluster"] = labels
    cluster_weather_labels = {}
    for cluster_id in range(c):
        cluster_summaries = numeric_data_sample[
            numeric_data_sample["cluster"] == cluster_id
        ]["summary"]
        most_common_summary = Counter(cluster_summaries).most_common(1)[0][0]
        if (
            "clear" in most_common_summary.lower()
            or "sun" in most_common_summary.lower()
        ):
            cluster_weather_labels[cluster_id] = "Sunny Weather"
        elif "cloud" in most_common_summary.lower():
            cluster_weather_labels[cluster_id] = "Cloudy"
        else:
            cluster_weather_labels[cluster_id] = "Others"
    numeric_data_sample["weather_label"] = numeric_data_sample["cluster"].map(
        cluster_weather_labels
    )
    weather_labels = ["Sunny Weather", "Cloudy Weather", "Others Weather"]
    # Plotting Figure
    plt.figure(figsize=(10, 6))
    for cluster_id in range(c):
        cluster_data = numeric_data_sample[numeric_data_sample["cluster"] == cluster_id]
        plt.scatter(
            cluster_data["use [kW]"],
            cluster_data["summary_encoded"],
            label=weather_labels[cluster_id],
            alpha=0.5,
        )

    plt.title("Clustering of HomeC Dataset Sample using Fuzzy C-Means")
    plt.xlabel("Using House Overall[kW] (Standardized)")
    plt.ylabel("Summary Encoded")
    plt.legend()
    save_plot(plt, GRAPH_FOLDER, "T5_1_fuzzy_cmean_clustering.png")
    # Save Metrics
    # Evaluation Metrics
    numeric_data_sample["predicted_label"] = numeric_data_sample["cluster"].map(
        cluster_weather_labels
    )
    numeric_data_sample["true_label"] = numeric_data_sample["summary"].apply(
        lambda x: (
            "Sunny Weather"
            if "clear" in x.lower() or "sun" in x.lower()
            else ("Cloudy" if "cloud" in x.lower() else "Others")
        )
    )

    # Encoding true and predicted labels numerically for evaluation
    label_mapping = {"Sunny Weather": 0, "Cloudy": 1, "Others": 2}
    numeric_data_sample["predicted_label_encoded"] = numeric_data_sample[
        "predicted_label"
    ].map(label_mapping)
    numeric_data_sample["true_label_encoded"] = numeric_data_sample["true_label"].map(
        label_mapping
    )

    # Extracting true and predicted labels
    y_true = numeric_data_sample["true_label_encoded"]
    y_pred = numeric_data_sample["predicted_label_encoded"]

    # Calculating the metrics
    calculate_and_save_metrics(y_true, y_pred, LOGS_FOLDER, "T5_1_fcmean_metrics.txt")


def hierarchical_clustering(data_pca, numeric_data_sample):
    """
    The following code was written using
    https://www.analyticsvidhya.com/blog/2024/05/understanding-fuzzy-c-means-clustering/
    """
    clustering = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(data_pca)
    hc_labels = clustering.labels_
    # Assign cluster labels to the dataset
    numeric_data_sample["hc_cluster"] = hc_labels
    # Plotting the dendrogram
    pairwise_distances = pdist(data_pca, metric="euclidean")
    linkage_matrix = hierarchy.linkage(pairwise_distances, method="ward")
    plt.figure(figsize=(12, 8))
    hierarchy.dendrogram(linkage_matrix)
    plt.title("Dendrogram for Hierarchical Clustering (Ward linkage)")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    save_plot(plt, GRAPH_FOLDER, "T5_2_hirarchy_dendogram_clustering.png")

    # Post-process clusters to determine weather type label for each cluster
    cluster_weather_labels_hc = {}
    for cluster_id in range(3):
        cluster_summaries = numeric_data_sample[
            numeric_data_sample["hc_cluster"] == cluster_id
        ]["summary"]
        if len(cluster_summaries) > 0:
            most_common_summary = Counter(cluster_summaries).most_common(1)[0][0]
            if (
                "clear" in most_common_summary.lower()
                or "sun" in most_common_summary.lower()
            ):
                cluster_weather_labels_hc[cluster_id] = "Sunny Weather"
            elif "cloud" in most_common_summary.lower():
                cluster_weather_labels_hc[cluster_id] = "Cloudy"
            else:
                cluster_weather_labels_hc[cluster_id] = "Others"
        else:
            cluster_weather_labels_hc[cluster_id] = "Others"

    numeric_data_sample["predicted_label"] = numeric_data_sample["hc_cluster"].map(
        cluster_weather_labels_hc
    )

    # Define true labels
    numeric_data_sample["true_label"] = numeric_data_sample["summary"].apply(
        lambda x: (
            "Sunny Weather"
            if "clear" in x.lower() or "sun" in x.lower()
            else ("Cloudy" if "cloud" in x.lower() else "Others")
        )
    )

    # Encode labels
    label_mapping = {"Sunny Weather": 0, "Cloudy": 1, "Others": 2}
    numeric_data_sample["predicted_label_encoded"] = numeric_data_sample[
        "predicted_label"
    ].map(label_mapping)
    numeric_data_sample["true_label_encoded"] = numeric_data_sample["true_label"].map(
        label_mapping
    )

    y_true = numeric_data_sample["true_label_encoded"]
    y_pred = numeric_data_sample["predicted_label_encoded"]
    # Calculating the metrics
    calculate_and_save_metrics(
        y_true, y_pred, LOGS_FOLDER, "T5_2_hirarchical_metrics.txt"
    )


def gaussian_mixture_clustering(data_pca, numeric_data_sample):
    """
    Perform Gaussian Mixture Model Clustering and evaluate results.
    https://brilliant.org/wiki/gaussian-mixture-model/
    """
    # Train Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42).fit(
        data_pca
    )
    gmm_labels = gmm.predict(data_pca)

    # Assign labels to the dataset
    numeric_data_sample["gmm_cluster"] = gmm_labels

    # Post-process clusters to determine weather type label for each cluster
    cluster_weather_labels_gmm = {}
    for cluster_id in range(3):
        cluster_summaries = numeric_data_sample[
            numeric_data_sample["gmm_cluster"] == cluster_id
        ]["summary"]
        if len(cluster_summaries) > 0:
            most_common_summary = Counter(cluster_summaries).most_common(1)[0][0]
            if (
                "clear" in most_common_summary.lower()
                or "sun" in most_common_summary.lower()
            ):
                cluster_weather_labels_gmm[cluster_id] = "Sunny Weather"
            elif "cloud" in most_common_summary.lower():
                cluster_weather_labels_gmm[cluster_id] = "Cloudy"
            else:
                cluster_weather_labels_gmm[cluster_id] = "Others"
        else:
            cluster_weather_labels_gmm[cluster_id] = "Others"

    numeric_data_sample["predicted_label"] = numeric_data_sample["gmm_cluster"].map(
        cluster_weather_labels_gmm
    )

    # Evaluation
    numeric_data_sample["true_label"] = numeric_data_sample["summary"].apply(
        lambda x: (
            "Sunny Weather"
            if "clear" in x.lower() or "sun" in x.lower()
            else ("Cloudy" if "cloud" in x.lower() else "Others")
        )
    )

    label_mapping = {"Sunny Weather": 0, "Cloudy": 1, "Others": 2}
    numeric_data_sample["predicted_label_encoded"] = numeric_data_sample[
        "predicted_label"
    ].map(label_mapping)
    numeric_data_sample["true_label_encoded"] = numeric_data_sample["true_label"].map(
        label_mapping
    )
    weather_labels = ["Sunny Weather", "Cloudy Weather", "Others Weather"]
    # Plotting Figure
    plt.figure(figsize=(10, 6))
    for cluster_id in range(3):
        cluster_data = numeric_data_sample[
            numeric_data_sample["gmm_cluster"] == cluster_id
        ]
        plt.scatter(
            cluster_data["use [kW]"],
            cluster_data["summary_encoded"],
            label=weather_labels[cluster_id],
            alpha=0.5,
        )

    plt.title("Clustering of HomeC Dataset Sample using Gaussian Mixture Clustering")
    plt.xlabel("Using House Overall[kW] (Standardized)")
    plt.ylabel("Summary Encoded")
    plt.legend()
    save_plot(plt, GRAPH_FOLDER, "T5_3_gaussian_mixture_clustering.png")

    y_true = numeric_data_sample["true_label_encoded"]
    y_pred = numeric_data_sample["predicted_label_encoded"]

    # Calculating the metrics
    calculate_and_save_metrics(
        y_true, y_pred, LOGS_FOLDER, "T5_3_gaussian_mixture_metrics.txt"
    )


def cluster_data_analysis(data):
    # Work with fuzzy C-means
    fuzzy_cmean_clustering(data)
    data_pca, numeric_data_sample = get_data_with_pca(data)
    # Hierarchical Clustering
    hierarchical_clustering(data_pca, numeric_data_sample)
    # Gaussian Mixture Clustering
    gaussian_mixture_clustering(data_pca, numeric_data_sample)
