#!/usr/bin/env python3
"""
Image Clustering: Data Processing and Parameter Optimization
--------------------------------------------------------------
Input: Directory "data/images", containing grayscale PNG images.
Output: Modified image datasets saved in directories "data/images_<strategy>",
        and an Excel file "results/image_clusters.xlsx" with clustering results.
Processing: Preprocess images and apply three modification strategies:
            - raw: Original image (no modifications).
            - threshold: Fixed thresholding for calcium; pixels with values in [193, 255]
                         become 255 (calcium), while all other pixels become 0.
            - combined: For each image, pixels are processed as follows:
                      • if the pixel value is in [193, 255], it is set to 255 (calcium);
                      • if the pixel value is in [1, 192], it is set to 124 (biological tissue);
                      • if the pixel value is 0, it remains unchanged.
            Then, clustering is performed with hyperparameter optimization using methods:
            KMeans, Gaussian Mixture Model (GMM), Spectral Clustering, Agglomerative Clustering, and OPTICS.
"""

import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def load_image(filepath):
    """Load an image in grayscale mode.

    Args:
        filepath (str): Path to the image file.

    Returns:
        tuple: A tuple (image, filename) where image is a numpy array of the grayscale image and filename is the base name.
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return image, os.path.basename(filepath)


def load_images(directory):
    """Load all PNG images from the specified directory using parallel processing.

    Args:
        directory (str): Directory containing PNG images.

    Returns:
        tuple: A tuple (images, image_names) where images is a list of loaded images and image_names is a list of filenames.
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


def image_preparing(images, max_width, max_height):
    """
    Preprocess images by converting to grayscale (if needed) and resizing them based on maximal allowed dimensions,
    while preserving the aspect ratio.

    Args:
        images (list of np.ndarray): List of images.
        max_width (int): Maximum allowed width.
        max_height (int): Maximum allowed height.

    Returns:
        list of np.ndarray: List of processed images.
    """
    processed_images = []
    for img in images:
        # Convert to grayscale if the image is in color.
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get original dimensions.
        orig_height, orig_width = img.shape[:2]

        # Compute the scale factor as the minimum of the ratios.
        scale = min(max_width / orig_width, max_height / orig_height)

        # Calculate new dimensions ensuring integer values.
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize the image using the computed dimensions.
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        processed_images.append(resized_img)

    return processed_images


def apply_image_modifications(images):
    """Generate modified versions of each image using three strategies.

    Strategies:
      - raw: Original image (no modifications).
      - threshold: Fixed thresholding; pixels with values in [193, 255] become 255, others 0.
      - combined:
          • if pixel in [193, 255] -> 255 (calcium);
          • if pixel in [1, 192] -> 124 (biological tissue);
          • if pixel == 0 -> remains 0.

    Args:
        images (list of np.ndarray): List of preprocessed grayscale images.

    Returns:
        dict: Dictionary mapping strategy names ("raw", "threshold", "combined") to lists of modified images.
    """
    mods = {}
    raw_images = images
    threshold_images = []
    combined_images = []

    for img in images:
        # Fixed thresholding: pixels in [193, 255] become 255; others become 0.
        _, thresh_img = cv2.threshold(img, 193, 255, cv2.THRESH_BINARY)
        threshold_images.append(thresh_img)

        # Combined modification: apply different mapping based on pixel values.
        combined_img = np.where(img >= 193, 255,
                                np.where((img >= 1) & (img <= 192), 124, 0)).astype(np.uint8)
        combined_images.append(combined_img)

    mods["raw"] = raw_images
    mods["threshold"] = threshold_images
    mods["combined"] = combined_images

    return mods


def store_modified_images(mods, image_names, base_dir="data"):
    """Save modified image datasets to subdirectories within the base directory.

    Each modification strategy is saved in a folder named "images_<strategy>".
    For the 'threshold' strategy, an Excel summary of white pixel counts is also saved.

    Args:
        mods (dict): Dictionary mapping strategy names to lists of modified images.
        image_names (list of str): List of original image filenames.
        base_dir (str, optional): Base directory to save modified images. Defaults to "data".

    Returns:
        None.
    """
    for mod_key, mod_images in mods.items():
        save_dir = os.path.join(base_dir, f"images_{mod_key}")
        os.makedirs(save_dir, exist_ok=True)

        # Initialize summary list for threshold strategy.
        threshold_summary = []

        for idx, img in enumerate(mod_images):
            filename = image_names[idx] if idx < len(image_names) else f"{mod_key}_{idx}.png"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, img)

            if mod_key == 'threshold':
                white_pixels = int(np.count_nonzero(img == 255))
                threshold_summary.append({
                    "Filename": filename,
                    "White_Pixels": white_pixels
                })

        if mod_key == 'threshold' and threshold_summary:
            df = pd.DataFrame(threshold_summary)
            excel_path = os.path.join(save_dir, "threshold_summary.xlsx")
            df.to_excel(excel_path, index=False)

        print(f"Saved {len(mod_images)} images for modification '{mod_key}' to {save_dir}")


def get_clustering_model(algorithm, params):
    """Instantiate and return a clustering model based on the algorithm name and parameters.

    Args:
        algorithm (str): Name of the clustering algorithm.
        params (dict): Parameters for the model.

    Returns:
        object: Clustering model instance.

    Raises:
        ValueError: If an unknown algorithm is provided.
    """
    if algorithm == "KMeans":
        return KMeans(n_clusters=params["n_clusters"],
                      n_init=params["n_init"], random_state=42)
    elif algorithm == "GMM":
        return GaussianMixture(n_components=params["n_components"],
                               covariance_type=params["covariance_type"],
                               random_state=42)
    elif algorithm == "Spectral":
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
        if params["linkage"] == "ward":
            return AgglomerativeClustering(n_clusters=params["n_clusters"],
                                           linkage=params["linkage"])
        else:
            return AgglomerativeClustering(n_clusters=params["n_clusters"],
                                           linkage=params["linkage"],
                                           metric=params["affinity"])
    elif algorithm == "OPTICS":
        return OPTICS(min_samples=params["min_samples"])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def cluster_with_params(features, algorithm, params):
    """Cluster the feature set using the specified algorithm and parameters.

    Args:
        features (np.ndarray): 2D array of flattened image features.
        algorithm (str): Clustering algorithm name.
        params (dict): Parameters for the clustering algorithm.

    Returns:
        np.ndarray or None: Array of cluster labels or None if clustering fails.
    """
    model = get_clustering_model(algorithm, params)
    try:
        if algorithm == "GMM":
            model.fit(features)
            labels = model.predict(features)
        elif algorithm == "OPTICS":
            model.fit(features)
            labels = model.labels_
        else:
            labels = model.fit_predict(features)
    except Exception as e:
        print(f"Error during clustering with {algorithm} and params {params}: {e}")
        labels = None
    return labels


def optimize_clustering(features, param_grids):
    """Iterate over algorithms and parameter combinations to perform clustering.

    Args:
        features (np.ndarray): 2D array of flattened image features.
        param_grids (dict): Dictionary mapping algorithm names to their parameter grids.

    Returns:
        list of dict: List of dictionaries containing clustering results for each configuration.
    """
    results_list = []
    for algorithm, grid in param_grids.items():
        keys = list(grid.keys())
        for values in product(*(grid[k] for k in keys)):
            params = dict(zip(keys, values))
            labels = cluster_with_params(features, algorithm, params)
            if labels is not None:
                unique_labels = np.unique(labels)
                actual_clusters = len(unique_labels)
            else:
                actual_clusters = None
            model_config = f"{algorithm}_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
            results_list.append({
                "model_config": model_config,
                "algorithm": algorithm,
                "params": params,
                "labels": labels,
                "actual_clusters": actual_clusters
            })
    return results_list


def save_assignments_excel(results_list, image_names, output_dir):
    """Save clustering results for all configurations to an Excel file.

    Each configuration name is prepended with a four-digit sequential number.

    Args:
        results_list (list of dict): List of clustering result dictionaries.
        image_names (list of str): List of image names.
        output_dir (str): Directory to save the Excel file.

    Returns:
        None.
    """
    os.makedirs(output_dir, exist_ok=True)
    assignments = {"Image": image_names}
    for res in results_list:
        model_config = res["model_config"]
        if res["labels"] is not None:
            assignments[model_config] = res["labels"].tolist()
        else:
            assignments[model_config] = [None] * len(image_names)
    df_assignments = pd.DataFrame(assignments)
    assignments_path = os.path.join(output_dir, "image_clusters.xlsx")
    df_assignments.to_excel(assignments_path, index=False)
    print(f"Saved clustering assignments to {assignments_path}")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """Main function to execute the image processing and clustering pipeline.

    Steps:
      1. Load and preprocess images.
      2. Apply three image modification strategies.
      3. Save modified image datasets.
      4. Perform clustering with hyperparameter optimization.
      5. Save clustering assignments to an Excel file.

    Returns:
        None.
    """
    # Parameters and directories.
    image_dir = "data/images"  # Input directory containing images.
    results_dir = "results"    # Output directory for clustering results.
    resized_width, resized_height = 64, 64
    clusters_number = [2, 3, 4, 5]  # Desired number of clusters.

    # Define parameter grids for each algorithm.
    param_grids = {
        "KMeans": {
            "n_clusters": clusters_number,
            "n_init": [5, 10, 15, 20]
        },
        "GMM": {
            "n_components": clusters_number,
            "covariance_type": ["full", "tied", "diag", "spherical"]
        },
        "Spectral": {
            "n_clusters": clusters_number,
            "affinity": ["nearest_neighbors", "rbf"],
            "n_neighbors": [5, 10, 15, 20]
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

    # Step 1: Load and preprocess images.
    print("Loading images...")
    images, image_names = load_images(image_dir)
    if not images:
        raise ValueError("No images loaded. Check the image directory path.")
    print("Preprocessing images...")
    processed_images = image_preparing(images, resized_width, resized_height)

    # Step 2: Apply image modification strategies.
    modifications = apply_image_modifications(processed_images)

    # Step 3: Save modified image datasets.
    store_modified_images(modifications, image_names, base_dir="data")

    # Container for clustering results.
    all_results_list = []

    # Step 4: Perform clustering for each modification strategy.
    for mod_key, mod_images in modifications.items():
        print(f"\nProcessing modification strategy: {mod_key}")
        features = np.array([img.flatten() for img in mod_images])
        print(f"Feature matrix shape for '{mod_key}': {features.shape}")

        # Optimize clustering over parameter grids.
        results_list = optimize_clustering(features, param_grids)
        for res in results_list:
            # Prepend modification strategy to the model configuration.
            res["model_config"] = f"{mod_key}_" + res["model_config"]
        all_results_list.extend(results_list)

    # Add unified numbering to clustering configuration names.
    for i, res in enumerate(all_results_list, start=1):
        res["model_config"] = f"{i:04d}_{res['model_config']}"

    # Step 5: Save clustering assignments.
    save_assignments_excel(all_results_list, image_names, results_dir)
    print("Clustering processing completed.")


if __name__ == "__main__":
    main()
