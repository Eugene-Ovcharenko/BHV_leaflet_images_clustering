#!/usr/bin/env python3
"""
Image Clustering with Parameter Optimization and Multi‐Output Charts
---------------------------------------------------------------------

This script loads a dataset of grayscale PNG images, preprocesses them,
and applies multiple clustering models with a grid search over hyper‐parameters.
The implemented algorithms are KMeans, Gaussian Mixture Model (GMM), Spectral
Clustering, Agglomerative Clustering, and OPTICS.
Deep optimization (i.e. broader parameter grids) is applied for Agglomerative,
GMM, and Spectral clustering. For algorithms that require a cluster number,
the script iterates over clusters_number = [2, 3] (where applicable).

In addition to clustering based on the original images, the code also creates
modified versions using several preprocessing strategies:
    - Thresholding Technique (Otsu’s thresholding)
    - Contrast Enhancement (histogram equalization)
    - Region of Interest (ROI) Extraction (using contour analysis)
    - Morphological Operations (opening to remove noise)
    - Feature Extraction Beyond Raw Intensities (edge detection via Canny)

For each modification strategy, evaluation metrics (Silhouette Score and
Calinski-Harabasz Index) are computed, and t‑SNE scatter plots are generated.
Projection images (AIP and MIP) are computed for each cluster and for the whole
dataset and are arranged in a grid (using subplots) in one figure per model configuration.

References:
    - van der Maaten, L. & Hinton, G. (2008). Visualizing Data using t-SNE.
    - Kaufman, L. & Rousseeuw, P. J. (1990). Finding Groups in Data: An Introduction to Cluster Analysis.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product
from concurrent.futures import ThreadPoolExecutor

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_image(filepath):
    """
    Load an image in grayscale.

    Parameters
    ----------
    filepath : str
        Path to the image file.

    Returns
    -------
    tuple
        (image: np.ndarray, filename: str)
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return image, os.path.basename(filepath)


def load_images(directory):
    """
    Load all PNG images from a given directory using parallel processing.

    Parameters
    ----------
    directory : str
        Directory containing PNG images.

    Returns
    -------
    tuple
        (list of images, list of image file names)
    """
    image_files = [os.path.join(directory, f) for f in os.listdir(directory)
                   if f.lower().endswith('.png')]
    images = []
    image_names = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_image, image_files))
        for img, name in results:
            if img is not None:
                images.append(img)
                image_names.append(name)
            else:
                print(f"Warning: Unable to load image {name}.")
    return images, image_names


def image_preparing(images, resized_width, resized_height):
    """
    Preprocess images by ensuring grayscale format and resizing.

    Parameters
    ----------
    images : list
        List of images (np.ndarray).
    resized_width : int
        Desired width.
    resized_height : int
        Desired height.

    Returns
    -------
    list
        List of processed images.
    """
    processed_images = []
    for img in images:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(img, (resized_width, resized_height),
                                 interpolation=cv2.INTER_AREA)
        processed_images.append(resized_img)
    return processed_images


# =============================================================================
# Image Modification Strategies
# =============================================================================

def apply_image_modifications(images):
    """
    Generate modified versions of each image using several strategies.

    Strategies include:
        - Raw (no modification)
        - Thresholded: Otsu’s thresholding
        - Contrast: Histogram equalization
        - ROI: Extract largest contour and crop/rescale
        - Morph: Morphological opening to remove noise
        - Edges: Canny edge detection (as a proxy for feature extraction)

    Parameters
    ----------
    images : list
        List of preprocessed images.

    Returns
    -------
    dict
        Dictionary mapping strategy names to lists of modified images.
    """
    mods = {}
    raw_images = images
    thresh_images = []
    contrast_images = []
    roi_images = []
    morph_images = []
    edge_images = []

    for img in images:
        # Thresholding using Otsu's method
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_images.append(thresh)

        # Contrast enhancement using histogram equalization
        contrast = cv2.equalizeHist(img)
        contrast_images.append(contrast)

        # ROI Extraction: find largest contour from the thresholded image,
        # extract bounding box, and resize back to original dimensions.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            roi = img[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi, (img.shape[1], img.shape[0]))
        else:
            roi_resized = img
        roi_images.append(roi_resized)

        # Morphological Operations: apply an opening operation to remove noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        morph_images.append(morph)

        # Feature Extraction Beyond Raw Intensities: apply Canny edge detection.
        edges = cv2.Canny(img, 100, 200)
        edge_images.append(edges)

    mods["raw"] = raw_images
    mods["thresholded"] = thresh_images
    mods["contrast"] = contrast_images
    mods["roi"] = roi_images
    mods["morph"] = morph_images
    mods["edges"] = edge_images
    return mods


# =============================================================================
# Clustering and Parameter Optimization
# =============================================================================

def get_clustering_model(algorithm, params):
    """
    Instantiate and return a clustering model based on the algorithm name and parameters.

    Parameters
    ----------
    algorithm : str
        Name of the clustering algorithm.
    params : dict
        Parameters for the model.

    Returns
    -------
    object
        Clustering model.
    """
    if algorithm == "KMeans":
        return KMeans(n_clusters=params["n_clusters"],
                      n_init=params["n_init"], random_state=42)
    elif algorithm == "GMM":
        return GaussianMixture(n_components=params["n_components"],
                               covariance_type=params["covariance_type"],
                               random_state=42)
    elif algorithm == "Spectral":
        # If using nearest_neighbors affinity, pass n_neighbors if provided.
        if params["affinity"] == "nearest_neighbors" and "n_neighbors" in params:
            return SpectralClustering(n_clusters=params["n_clusters"],
                                      affinity=params["affinity"],
                                      n_neighbors=params["n_neighbors"],
                                      assign_labels="discretize",
                                      random_state=42)
        else:
            return SpectralClustering(n_clusters=params["n_clusters"],
                                      affinity=params["affinity"],
                                      assign_labels="discretize",
                                      random_state=42)
    elif algorithm == "Agglomerative":
        # For 'ward' linkage, metric is fixed to Euclidean and should not be set.
        if params["linkage"] == "ward":
            return AgglomerativeClustering(n_clusters=params["n_clusters"],
                                           linkage=params["linkage"])
        else:
            # In recent versions of scikit-learn, 'affinity' is renamed to 'metric'
            return AgglomerativeClustering(n_clusters=params["n_clusters"],
                                           linkage=params["linkage"],
                                           metric=params["affinity"])
    elif algorithm == "OPTICS":
        return OPTICS(min_samples=params["min_samples"])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def cluster_with_params(features, algorithm, params):
    """
    Cluster the features using a specific algorithm and parameters.

    Parameters
    ----------
    features : np.ndarray
        Flattened image features.
    algorithm : str
        Clustering algorithm name.
    params : dict
        Parameters for the clustering algorithm.

    Returns
    -------
    np.ndarray
        Array of cluster labels.
    """
    model = get_clustering_model(algorithm, params)
    try:
        if algorithm == "GMM":
            model.fit(features)
            labels = model.predict(features)
        elif algorithm in ["OPTICS"]:
            model.fit(features)
            labels = model.labels_
        else:
            labels = model.fit_predict(features)
    except Exception as e:
        print(f"Error during clustering with {algorithm} and params {params}: {e}")
        labels = None
    return labels


def optimize_clustering(features, param_grids):
    """
    Iterate over algorithms and parameter combinations.

    Parameters
    ----------
    features : np.ndarray
        2D array of flattened images.
    param_grids : dict
        Dictionary mapping algorithm names to their parameter grids.

    Returns
    -------
    list
        List of dictionaries containing results for each model configuration.
    """
    results_list = []
    for algorithm, grid in param_grids.items():
        keys = list(grid.keys())
        for values in product(*(grid[k] for k in keys)):
            params = dict(zip(keys, values))
            labels = cluster_with_params(features, algorithm, params)
            if labels is not None:
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1 and len(unique_labels) < features.shape[0]:
                    sil_score = silhouette_score(features, labels)
                    ch_score = calinski_harabasz_score(features, labels)
                else:
                    sil_score = np.nan
                    ch_score = np.nan
                actual_clusters = len(unique_labels)
            else:
                sil_score = np.nan
                ch_score = np.nan
                actual_clusters = np.nan
            model_config = f"{algorithm}_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
            results_list.append({
                "model_config": model_config,
                "algorithm": algorithm,
                "params": params,
                "labels": labels,
                "silhouette_score": sil_score,
                "calinski_harabasz_score": ch_score,
                "actual_clusters": actual_clusters
            })
    return results_list


# =============================================================================
# Visualization: t-SNE and Projection Plots
# =============================================================================

def visualize_tsne_config(tsne_results, image_names, labels, model_config, output_dir):
    """
    Create and save a t-SNE scatter plot and underlying data in Excel.

    Parameters
    ----------
    tsne_results : np.ndarray
        2D t-SNE coordinates.
    image_names : list
        List of image file names.
    labels : np.ndarray
        Cluster labels.
    model_config : str
        Identifier for the model configuration.
    output_dir : str
        Base output directory.
    """
    tsne_dir = os.path.join(output_dir, "tsne_plots")
    os.makedirs(tsne_dir, exist_ok=True)
    tsne_data_dir = os.path.join(output_dir, "tsne_data")
    os.makedirs(tsne_data_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1],
                    hue=labels, palette="viridis", legend="full")
    plt.title(f"t-SNE: {model_config}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Cluster")
    plot_filename = os.path.join(tsne_dir, f"{model_config}_tsne.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot for {model_config} at {plot_filename}")

    # Save the underlying t-SNE data in Excel.
    df_tsne = pd.DataFrame({
        "Image": image_names,
        "tsne1": tsne_results[:, 0],
        "tsne2": tsne_results[:, 1],
        "Cluster": labels
    })
    tsne_excel_path = os.path.join(tsne_data_dir, f"{model_config}_tsne.xlsx")
    df_tsne.to_excel(tsne_excel_path, index=False)
    print(f"Saved t-SNE data for {model_config} at {tsne_excel_path}")


def compute_projection(images, mode="MIP"):
    """
    Compute Maximum or Average Intensity Projection from a list of images.

    Parameters
    ----------
    images : list
        List of images (np.ndarray).
    mode : str, optional
        'MIP' for maximum intensity or 'AIP' for average intensity (default: 'MIP').

    Returns
    -------
    np.ndarray
        The projected image.
    """
    if not images:
        return None
    stack = np.stack(images, axis=0)
    if mode.upper() == "MIP":
        return np.max(stack, axis=0)
    elif mode.upper() == "AIP":
        return np.mean(stack, axis=0)
    else:
        raise ValueError("Mode must be either 'MIP' or 'AIP'")


def visualize_projection_config(processed_images, labels, model_config, output_dir):
    """
    Create a combined figure (grid of subplots) with AIP and MIP images per cluster.
    Save the figure (PNG) without saving the underlying numerical data.

    Parameters
    ----------
    processed_images : list
        List of preprocessed (or modified) images.
    labels : np.ndarray
        Cluster labels.
    model_config : str
        Identifier for the model configuration.
    output_dir : str
        Base output directory.
    """
    proj_plot_dir = os.path.join(output_dir, "projection_plots")
    os.makedirs(proj_plot_dir, exist_ok=True)

    unique_labels = np.unique(labels)
    if labels is None or len(unique_labels) == 0:
        return
    num_clusters = len(unique_labels)
    # Create grid with 2 rows: row 0 = AIP, row 1 = MIP; add one extra column for overall projections.
    cols = num_clusters + 1
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))

    # Compute overall projections.
    overall_aip = compute_projection(processed_images, mode="AIP")
    overall_mip = compute_projection(processed_images, mode="MIP")

    # Plot overall projections in the last column.
    axes[0, -1].imshow(overall_aip, cmap="gray")
    axes[0, -1].set_title("Overall AIP")
    axes[0, -1].axis("off")
    axes[1, -1].imshow(overall_mip, cmap="gray")
    axes[1, -1].set_title("Overall MIP")
    axes[1, -1].axis("off")

    for i, cluster in enumerate(unique_labels):
        cluster_imgs = [img for img, lab in zip(processed_images, labels) if lab == cluster]
        aip = compute_projection(cluster_imgs, mode="AIP")
        mip = compute_projection(cluster_imgs, mode="MIP")
        axes[0, i].imshow(aip, cmap="gray")
        axes[0, i].set_title(f"Cluster {cluster} AIP")
        axes[0, i].axis("off")
        axes[1, i].imshow(mip, cmap="gray")
        axes[1, i].set_title(f"Cluster {cluster} MIP")
        axes[1, i].axis("off")

    fig.suptitle(f"Projections for {model_config}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    proj_plot_filename = os.path.join(proj_plot_dir, f"{model_config}_projections.png")
    plt.savefig(proj_plot_filename, dpi=300)
    plt.close()
    print(f"Saved projection plot for {model_config} at {proj_plot_filename}")


# =============================================================================
# Save Overall Results (Clustering Assignments and Metrics)
# =============================================================================

def save_all_results_excel(results_list, image_names, output_dir):
    """
    Save clustering assignments and metrics for all model configurations in Excel files.

    Parameters
    ----------
    results_list : list
        List of dictionaries with clustering results.
    image_names : list
        List of image file names.
    output_dir : str
        Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prepare clustering assignments (one column per model configuration).
    assignments = {"Image": image_names}
    for res in results_list:
        model_config = res["model_config"]
        if res["labels"] is not None:
            assignments[model_config] = res["labels"]
        else:
            assignments[model_config] = [np.nan] * len(image_names)
    df_assignments = pd.DataFrame(assignments)
    assignments_path = os.path.join(output_dir, "image_clusters.xlsx")
    df_assignments.to_excel(assignments_path, index=False)
    print(f"Saved clustering assignments to {assignments_path}")

    # Prepare metrics for each model configuration.
    metrics_data = []
    for res in results_list:
        exp_clusters = res["params"].get("n_clusters",
                                         res["params"].get("n_components", "N/A"))
        metrics_data.append({
            "Model_Config": res["model_config"],
            "Algorithm": res["algorithm"],
            "Parameters": str(res["params"]),
            "Expected_n_clusters": exp_clusters,
            "Actual_Clusters": res["actual_clusters"],
            "Silhouette_Score": res["silhouette_score"],
            "Calinski_Harabasz_Score": res["calinski_harabasz_score"]
        })
    df_metrics = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(output_dir, "cluster_metrics.xlsx")
    df_metrics.to_excel(metrics_path, index=False)
    print(f"Saved clustering metrics to {metrics_path}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    # Parameters and directories.
    image_dir = "data/images"  # Directory where images are stored.
    results_dir = "results"
    resized_width, resized_height = 64, 64
    clusters_number = [2, 3]  # For algorithms that require a cluster number.

    # Define parameter grids for each algorithm.
    # Note: Deeper optimization is provided for Agglomerative, GMM, and Spectral.
    param_grids = {
        "KMeans": {
            "n_clusters": clusters_number,
            "n_init": [10, 20]
        },
        "GMM": {
            "n_components": clusters_number,
            "covariance_type": ["full", "tied", "diag", "spherical"]
        },
        "Spectral": {
            "n_clusters": clusters_number,
            "affinity": ["nearest_neighbors", "rbf"],
            "n_neighbors": [5, 10, 15]  # Only used if affinity=='nearest_neighbors'
        },
        "Agglomerative": {
            "n_clusters": clusters_number,
            "linkage": ["ward", "complete", "average", "single"],
            "affinity": ["euclidean", "manhattan", "cosine"]
        },
        "OPTICS": {
            "min_samples": [3, 5, 10]
        }
    }

    # Step 1: Data Loading & Preprocessing.
    print("Loading images...")
    images, image_names = load_images(image_dir)
    if not images:
        raise ValueError("No images loaded. Check the image directory path.")
    print("Preprocessing images...")
    processed_images = image_preparing(images, resized_width, resized_height)
    print(f"Feature matrix shape: {np.array(processed_images).shape}")

    # Apply modification strategies.
    modifications = apply_image_modifications(processed_images)

    # Container for collecting all clustering results.
    all_results_list = []

    # Process each modification strategy (including "raw").
    for mod_key, mod_images in modifications.items():
        print(f"\nProcessing modification strategy: {mod_key}")
        # Compute features by flattening each image variant.
        features = np.array([img.flatten() for img in mod_images])
        print(f"Feature matrix shape for {mod_key}: {features.shape}")

        # Compute t-SNE on the features.
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(features)

        # Parameter Optimization & Clustering.
        print("Performing clustering with parameter optimization...")
        results_list = optimize_clustering(features, param_grids)

        # Append modification identifier to each model configuration.
        for res in results_list:
            res["model_config"] = f"{mod_key}_" + res["model_config"]
            # Visualization for each model configuration.
            print(f"Visualizing results for {res['model_config']} ...")
            visualize_tsne_config(tsne_results, image_names, res["labels"],
                                  res["model_config"], results_dir)
            visualize_projection_config(mod_images, res["labels"],
                                        res["model_config"], results_dir)

        all_results_list.extend(results_list)

    # Save Overall Clustering Assignments and Metrics.
    save_all_results_excel(all_results_list, image_names, results_dir)
    print("Processing completed.")


if __name__ == "__main__":
    main()
