#!/usr/bin/env python3
"""
Image Clustering with Parameter Optimization and Multi‐Output Charts
---------------------------------------------------------------------

This script reads clustering results from "results/image_clusters.xlsx" and calculates
additional metrics such as Davies-Bouldin Index, WCSS, Dunn Index, and the percentage
distribution of images in each cluster. It also creates t-SNE and PCA scatter plots
and projection plots.

Each function responsible for metric computation or visualization accepts a single data
source (a directory of images) and internally loads the images and their vector representations
via a dedicated loader function.
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,
    'mathtext.default': 'regular'
})

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def safe_imshow(ax, img, title):
    """Safely display an image on the given axis.

    If the image is empty or None, display a "No data" title and turn off the axis.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to display the image.
        img (array-like): The image data.
        title (str): The title for the axis.

    Returns:
        None.
    """
    if img is None or np.asarray(img).size == 0:
        ax.set_title(f"{title}\n(No data)")
        ax.axis("off")
        return
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def get_axis_limits(data, margin_ratio=0.1):
    """Calculate 'nice' axis limits based on the data range with a margin.

    Args:
        data (array-like): The data from which to calculate limits.
        margin_ratio (float, optional): Fractional margin to add. Defaults to 0.1.

    Returns:
        tuple: A tuple (lower, upper, base) where lower and upper are the adjusted limits,
               and base is the scaling factor used.
    """
    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min

    if data_range == 0:
        return data_min - 1, data_max + 1, 1

    base = 10 ** int(np.floor(np.log10(abs(data_range))))
    margin = margin_ratio * data_range
    adj_min = data_min - margin
    adj_max = data_max + margin
    lower = base * np.floor(adj_min / base)
    upper = base * np.ceil(adj_max / base)
    return lower, upper, base


def comma_formatter(x, pos):
    """Format a number by replacing the decimal point with a comma.

    Args:
        x (float): The number to format.
        pos (int): Tick position (unused).

    Returns:
        str: The formatted number with a comma as the decimal separator.
    """
    s = f"{x:.1f}"
    return s.replace('.', ',')


def int_formatter(x, pos):
    """Format a number as an integer string.

    Args:
        x (float): The number to format.
        pos (int): Tick position (unused).

    Returns:
        str: The integer as a string.
    """
    return f"{int(round(x))}"


def load_image(filepath):
    """Load an image in grayscale and return the image and its basename.

    Args:
        filepath (str): Path to the image file.

    Returns:
        tuple: A tuple (image, basename) where image is the loaded grayscale image and basename is the file name.
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return image, os.path.basename(filepath)


def load_images(directory):
    """Load all PNG images from a given directory using parallel processing.

    Args:
        directory (str): Directory containing PNG images.

    Returns:
        tuple: A tuple (images, image_names) where images is a list of loaded images and
               image_names is a list of corresponding file names.
    """
    image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory)
                          if f.lower().endswith('.png')])
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


def load_data(source):
    """Load images and extract flattened features from the given source directory.

    Args:
        source (str): Directory path containing the images.

    Returns:
        tuple: A tuple (images, image_names, features) where:
            - images is a list of loaded grayscale images,
            - image_names is a list of image file names,
            - features is a numpy array of flattened image representations.
    """
    images, image_names = load_images(source)
    features = np.array([img.flatten() for img in images])
    return images, image_names, features

# -----------------------------------------------------------------------------
# Principal (Core) Functions: Visualization & Metrics Computation
# -----------------------------------------------------------------------------

def visualize_cluster_images_with_names(data_source, labels, model_config, output_dir):
    """Visualize images grouped by cluster with their corresponding names.

    Args:
        data_source (str): Directory path containing the images.
        labels (array-like): Cluster labels for each image.
        model_config (str): Configuration string used for titling and saving output.
        output_dir (str): Directory to save the resulting plot.

    Returns:
        None.
    """
    images, image_names, _ = load_data(data_source)
    # Group images and names by cluster
    cluster_dict = {}
    for img, name, lab in zip(images, image_names, labels):
        cluster_dict.setdefault(lab, []).append((img, name))
    clusters = sorted(cluster_dict.keys())
    n_clusters = len(clusters)
    max_imgs = max(len(cluster_dict[cl]) for cl in clusters)
    fig_width = 3 * n_clusters
    fig_height = 3 * max_imgs
    fig, axes = plt.subplots(max_imgs, n_clusters, figsize=(fig_width, fig_height))
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        if max_imgs == 1:
            axes = axes[np.newaxis, :]
        else:
            axes = axes[:, np.newaxis]
    for col_idx, cl in enumerate(clusters):
        imgs_in_cluster = cluster_dict[cl]
        n_imgs = len(imgs_in_cluster)
        for row_idx in range(max_imgs):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            if row_idx < n_imgs:
                img, name = imgs_in_cluster[row_idx]
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
                ax.text(0.5, -0.08, name, transform=ax.transAxes,
                        ha='center', va='top', fontsize=8, color='black',
                        bbox=dict(facecolor='white', edgecolor='none', pad=1))
        axes[0, col_idx].set_title(f"Cluster {cl}", fontsize=12)
    fig.suptitle(f"Images by Clusters: {model_config}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    cluster_img_dir = os.path.join(output_dir, "cluster_images")
    os.makedirs(cluster_img_dir, exist_ok=True)
    plot_filename = os.path.join(cluster_img_dir, f"{model_config}_cluster_images_with_names.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Saved cluster images with names plot for {model_config} at {plot_filename}")


def compute_wcss(data_source, labels):
    """Compute the Within-Cluster Sum of Squares (WCSS) for vectorized images.

    Args:
        data_source (str): Directory path containing the images.
        labels (array-like): Cluster labels.

    Returns:
        float: The computed WCSS value.
    """
    _, _, features = load_data(data_source)
    total_wcss = 0.0
    for cl in np.unique(labels):
        cluster_points = features[labels == cl]
        centroid = np.mean(cluster_points, axis=0)
        total_wcss += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
    return total_wcss


def compute_dunn_index(data_source, labels):
    """Compute the Dunn Index.

    Dunn Index = (minimum inter-cluster distance) / (maximum intra-cluster distance)

    Args:
        data_source (str): Directory path containing the images.
        labels (array-like): Cluster labels.

    Returns:
        float: The computed Dunn Index, or NaN if not computable.
    """
    _, _, features = load_data(data_source)
    clusters = np.unique(labels)
    if len(clusters) < 2:
        return np.nan
    intra_dists = []
    for cl in clusters:
        cluster_points = features[labels == cl]
        if len(cluster_points) < 2:
            intra_dists.append(0)
        else:
            dists = np.linalg.norm(cluster_points[:, None] - cluster_points, axis=2)
            intra_dists.append(np.max(dists))
    max_intra = np.max(intra_dists)
    inter_dists = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            points_i = features[labels == clusters[i]]
            points_j = features[labels == clusters[j]]
            dists = np.linalg.norm(points_i[:, None] - points_j, axis=2)
            inter_dists.append(np.min(dists))
    min_inter = np.min(inter_dists)
    return min_inter / max_intra if max_intra > 0 else np.nan


def visualize_tsne_config(data_source, labels, model_config, output_dir):
    """Compute t-SNE on vectorized images and create a scatter plot with cluster information.

    Args:
        data_source (str): Directory path containing the images.
        labels (array-like): Cluster labels.
        model_config (str): Configuration string for titling and saving output.
        output_dir (str): Directory to save the t-SNE plot and data.

    Returns:
        None.
    """
    _, image_names, features = load_data(data_source)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    tsne_dir = os.path.join(output_dir, "tsne_plots")
    os.makedirs(tsne_dir, exist_ok=True)
    tsne_data_dir = os.path.join(output_dir, "tsne_data")
    os.makedirs(tsne_data_dir, exist_ok=True)
    sns.set_style("whitegrid")
    original_labels = np.array(labels)
    offset = original_labels.min() if original_labels.min() < 0 else 0
    adjusted_labels = original_labels - offset
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(comma_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(comma_formatter))
    scatter = sns.scatterplot(x=tsne_results[:, 0],
                              y=tsne_results[:, 1],
                              hue=adjusted_labels,
                              palette="tab10",
                              legend="full")
    plt.title(f"t-SNE: {model_config}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    x_lower, x_upper, base_x = get_axis_limits(tsne_results[:, 0])
    y_lower, y_upper, base_y = get_axis_limits(tsne_results[:, 1])
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xticks(np.arange(x_lower, x_upper + base_x, base_x))
    ax.set_yticks(np.arange(y_lower, y_upper + base_y, base_y))
    unique_adjusted = sorted(np.unique(adjusted_labels))
    unique_original = [int(val + offset) for val in unique_adjusted]
    palette = sns.color_palette("tab10", n_colors=len(unique_adjusted))
    legend_handles = [Line2D([0], [0], marker='o', color='w', label=str(orig),
                             markerfacecolor=palette[i], markersize=8)
                      for i, orig in enumerate(unique_original)]
    ax.legend(handles=legend_handles, title="Cluster")
    plot_filename = os.path.join(tsne_dir, f"{model_config}_tsne.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot for {model_config} at {plot_filename}")
    df_tsne = pd.DataFrame({
        "Image": image_names,
        "tsne1": tsne_results[:, 0],
        "tsne2": tsne_results[:, 1],
        "Cluster": original_labels
    })
    tsne_excel_path = os.path.join(tsne_data_dir, f"{model_config}_tsne.xlsx")
    df_tsne.to_excel(tsne_excel_path, index=False)
    print(f"Saved t-SNE data for {model_config} at {tsne_excel_path}")


def visualize_tsne_with_images(data_source, labels, model_config, output_dir):
    """Compute t-SNE and create a scatter plot with image thumbnails.

    Args:
        data_source (str): Directory path containing the images.
        labels (array-like): Cluster labels.
        model_config (str): Configuration string for titling and saving output.
        output_dir (str): Directory to save the t-SNE plot with images and data.

    Returns:
        None.
    """
    images, image_names, features = load_data(data_source)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    original_labels = np.array(labels)
    offset = original_labels.min() if original_labels.min() < 0 else 0
    adjusted_labels = original_labels - offset
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(comma_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(comma_formatter))
    x_lower, x_upper, base_x = get_axis_limits(tsne_results[:, 0])
    y_lower, y_upper, base_y = get_axis_limits(tsne_results[:, 1])
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xticks(np.arange(x_lower, x_upper + base_x, base_x))
    ax.set_yticks(np.arange(y_lower, y_upper + base_y, base_y))
    unique_adjusted = sorted(np.unique(adjusted_labels))
    unique_original = [int(val + offset) for val in unique_adjusted]
    palette_colors = sns.color_palette("tab10", n_colors=len(unique_adjusted))

    def get_cluster_color(label):
        idx = unique_adjusted.index(label)
        return palette_colors[idx]

    for i, (x, y) in enumerate(tsne_results):
        img = images[i]
        oi = OffsetImage(img, cmap='gray', zoom=0.5, norm=plt.Normalize(0, 255))
        cluster_color = get_cluster_color(adjusted_labels[i])
        ab = AnnotationBbox(oi, (x, y), frameon=True,
                            bboxprops=dict(edgecolor=cluster_color, linewidth=1))
        ax.add_artist(ab)
    plt.title(f"t-SNE with Images: {model_config}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    legend_handles = [Line2D([0], [0], marker='o', color='w', label=str(orig),
                             markerfacecolor=palette_colors[i], markersize=8)
                      for i, orig in enumerate(unique_original)]
    ax.legend(handles=legend_handles, title="Cluster")
    tsne_img_dir = os.path.join(output_dir, "tsne_images_plots")
    os.makedirs(tsne_img_dir, exist_ok=True)
    plot_filename = os.path.join(tsne_img_dir, f"{model_config}_tsne_with_images.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Saved t-SNE with images plot for {model_config} at {plot_filename}")
    df_tsne = pd.DataFrame({
        "Image": image_names,
        "tsne1": tsne_results[:, 0],
        "tsne2": tsne_results[:, 1],
        "Cluster": original_labels
    })
    tsne_excel_dir = os.path.join(output_dir, "tsne_images_data")
    os.makedirs(tsne_excel_dir, exist_ok=True)
    tsne_excel_path = os.path.join(tsne_excel_dir, f"{model_config}_tsne_with_images.xlsx")
    df_tsne.to_excel(tsne_excel_path, index=False)
    print(f"Saved t-SNE with images data for {model_config} at {tsne_excel_path}")


def visualize_pca_config(data_source, labels, model_config, output_dir):
    """Compute PCA on vectorized images and create a scatter plot with cluster information.

    Args:
        data_source (str): Directory path containing the images.
        labels (array-like): Cluster labels.
        model_config (str): Configuration string for titling and saving output.
        output_dir (str): Directory to save the PCA plot and data.

    Returns:
        None.
    """
    _, image_names, features = load_data(data_source)
    pca_dir = os.path.join(output_dir, "pca_plots")
    os.makedirs(pca_dir, exist_ok=True)
    pca_data_dir = os.path.join(output_dir, "pca_data")
    os.makedirs(pca_data_dir, exist_ok=True)
    sns.set_style("whitegrid")
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(features)
    original_labels = np.array(labels)
    offset = original_labels.min() if original_labels.min() < 0 else 0
    adjusted_labels = original_labels - offset
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(int_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(int_formatter))
    scatter = sns.scatterplot(x=pca_results[:, 0],
                              y=pca_results[:, 1],
                              hue=adjusted_labels,
                              palette="tab10",
                              legend="full")
    plt.title(f"PCA: {model_config}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    x_lower, x_upper, base_x = get_axis_limits(pca_results[:, 0])
    y_lower, y_upper, base_y = get_axis_limits(pca_results[:, 1])
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xticks(np.arange(x_lower, x_upper + base_x, base_x))
    ax.set_yticks(np.arange(y_lower, y_upper + base_y, base_y))
    unique_adjusted = sorted(np.unique(adjusted_labels))
    unique_original = [int(val + offset) for val in unique_adjusted]
    palette = sns.color_palette("tab10", n_colors=len(unique_adjusted))
    legend_handles = [Line2D([0], [0], marker='o', color='w', label=str(orig),
                             markerfacecolor=palette[i], markersize=8)
                      for i, orig in enumerate(unique_original)]
    ax.legend(handles=legend_handles, title="Cluster")
    plot_filename = os.path.join(pca_dir, f"{model_config}_pca.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Saved PCA plot for {model_config} at {plot_filename}")
    df_pca = pd.DataFrame({
        "Image": image_names,
        "PC1": pca_results[:, 0],
        "PC2": pca_results[:, 1],
        "Cluster": original_labels
    })
    pca_excel_path = os.path.join(pca_data_dir, f"{model_config}_pca.xlsx")
    df_pca.to_excel(pca_excel_path, index=False)
    print(f"Saved PCA data for {model_config} at {pca_excel_path}")


def visualize_pca_with_images(data_source, labels, model_config, output_dir):
    """Compute PCA on vectorized images and create a scatter plot with image thumbnails.

    Args:
        data_source (str): Directory path containing the images.
        labels (array-like): Cluster labels.
        model_config (str): Configuration string for titling and saving output.
        output_dir (str): Directory to save the PCA plot with images and data.

    Returns:
        None.
    """
    images, image_names, features = load_data(data_source)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(features)
    original_labels = np.array(labels)
    offset = original_labels.min() if original_labels.min() < 0 else 0
    adjusted_labels = original_labels - offset
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(int_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(int_formatter))
    x_lower, x_upper, base_x = get_axis_limits(pca_results[:, 0])
    y_lower, y_upper, base_y = get_axis_limits(pca_results[:, 1])
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xticks(np.arange(x_lower, x_upper + base_x, base_x))
    ax.set_yticks(np.arange(y_lower, y_upper + base_y, base_y))
    unique_adjusted = sorted(np.unique(adjusted_labels))
    unique_original = [int(val + offset) for val in unique_adjusted]
    palette_colors = sns.color_palette("tab10", n_colors=len(unique_adjusted))

    def get_cluster_color(label):
        idx = unique_adjusted.index(label)
        return palette_colors[idx]

    for i, (x, y) in enumerate(pca_results):
        img = images[i]
        oi = OffsetImage(img, cmap='gray', zoom=0.5, norm=plt.Normalize(0, 255))
        cluster_color = get_cluster_color(adjusted_labels[i])
        ab = AnnotationBbox(oi, (x, y), frameon=True,
                            bboxprops=dict(edgecolor=cluster_color, linewidth=1))
        ax.add_artist(ab)
    plt.title(f"PCA with Images: {model_config}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    legend_handles = [Line2D([0], [0], marker='o', color='w', label=str(orig),
                             markerfacecolor=palette_colors[i], markersize=8)
                      for i, orig in enumerate(unique_original)]
    ax.legend(handles=legend_handles, title="Cluster")
    pca_img_dir = os.path.join(output_dir, "pca_images_plots")
    os.makedirs(pca_img_dir, exist_ok=True)
    plot_filename = os.path.join(pca_img_dir, f"{model_config}_pca_with_images.png")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Saved PCA with images plot for {model_config} at {plot_filename}")
    df_pca = pd.DataFrame({
        "Image": image_names,
        "PC1": pca_results[:, 0],
        "PC2": pca_results[:, 1],
        "Cluster": original_labels
    })
    pca_excel_dir = os.path.join(output_dir, "pca_images_data")
    os.makedirs(pca_excel_dir, exist_ok=True)
    pca_excel_path = os.path.join(pca_excel_dir, f"{model_config}_pca_with_images.xlsx")
    df_pca.to_excel(pca_excel_path, index=False)
    print(f"Saved PCA with images data for {model_config} at {pca_excel_path}")


def compute_projection(images, mode="MIP"):
    """Compute the Maximum Intensity Projection (MIP) from a list of images.

    Args:
        images (list[np.ndarray]): List of images.
        mode (str, optional): Projection mode. Currently only 'MIP' is supported. Defaults to "MIP".

    Returns:
        np.ndarray or None: The computed projection image, or None if the image list is empty.

    Raises:
        ValueError: If the specified mode is not "MIP".
    """
    if not images:
        return None
    if mode.upper() != "MIP":
        raise ValueError("Only MIP mode is supported in this version.")
    stack = np.stack(images, axis=0)
    return np.max(stack, axis=0)


def visualize_projection_config(data_source, labels, model_config, output_dir):
    """Create and save a projection plot for the given configuration.

    Depending on the configuration key, the function either computes projections
    on processed images only or combines projections from both processed and raw images.

    Args:
        data_source (str): Directory path containing processed images.
        labels (array-like): Cluster labels.
        model_config (str): Configuration string for titling and file naming.
        output_dir (str): Directory to save the projection plot.

    Returns:
        None.
    """
    proj_plot_dir = os.path.join(output_dir, "projection_plots")
    os.makedirs(proj_plot_dir, exist_ok=True)
    unique_labels = np.unique(labels)
    if labels is None or len(unique_labels) == 0:
        return
    mod_key = model_config.split("_")[0].lower()

    images, _, _ = load_data(data_source)
    if mod_key == "raw":
        cols = len(unique_labels) + 1
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
        overall_mip = compute_projection(images, mode="MIP")
        safe_imshow(axes[-1], overall_mip, "Overall MIP")
        for i, cluster in enumerate(unique_labels):
            cluster_imgs = [img for img, lab in zip(images, labels) if lab == cluster]
            mip = compute_projection(cluster_imgs, mode="MIP")
            safe_imshow(axes[i], mip, f"Cluster {cluster} MIP")
        fig.suptitle(f"Projections for {model_config}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        proj_plot_filename = os.path.join(proj_plot_dir, f"{model_config}_projections.png")
        plt.savefig(proj_plot_filename, dpi=300)
        plt.close()
        print(f"Saved projection plot for {model_config} at {proj_plot_filename}")
    else:
        raw_folder = os.path.join("data", "raw")
        if not os.path.exists(raw_folder):
            raw_folder = os.path.join("data", "images_raw")
        _, _, _ = load_data(data_source)  # processed images already loaded above
        # Load raw images using the same loader for consistency.
        raw_images, _, _ = load_data(raw_folder)
        cols = len(unique_labels) + 1
        fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))
        overall_mip_mod = compute_projection(images, mode="MIP")
        safe_imshow(axes[0, -1], overall_mip_mod, "Overall MIP (modified)")
        for i, cluster in enumerate(unique_labels):
            cluster_imgs_mod = [img for img, lab in zip(images, labels) if lab == cluster]
            mip_mod = compute_projection(cluster_imgs_mod, mode="MIP")
            safe_imshow(axes[0, i], mip_mod, f"Cluster {cluster} MIP (modified)")
        overall_mip_raw = compute_projection(raw_images, mode="MIP")
        safe_imshow(axes[1, -1], overall_mip_raw, "Overall MIP (raw)")
        for i, cluster in enumerate(unique_labels):
            cluster_imgs_raw = [img for img, lab in zip(raw_images, labels) if lab == cluster]
            mip_raw = compute_projection(cluster_imgs_raw, mode="MIP")
            safe_imshow(axes[1, i], mip_raw, f"Cluster {cluster} MIP (raw)")
        fig.suptitle(f"Projections for {model_config}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        proj_plot_filename = os.path.join(proj_plot_dir, f"{model_config}_projections.png")
        plt.savefig(proj_plot_filename, dpi=300)
        plt.close()
        print(f"Saved projection plot for {model_config} at {proj_plot_filename}")


def save_all_results_excel(results_list, image_names, output_dir):
    """Save clustering metrics for all model configurations in an Excel file.

    Args:
        results_list (list[dict]): List of dictionaries containing clustering results.
        image_names (list[str]): List of image names.
        output_dir (str): Directory to save the Excel file.

    Returns:
        None.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_data = []
    for res in results_list:
        model_config = res["model_config"]
        tokens = model_config.split("_", 2)
        if len(tokens) >= 3:
            file_id = tokens[0]
            image_processing = tokens[1]
            algorithm = tokens[2]
            parameters = tokens[2]
            m_expected = re.search(r'(n_clusters|n_components)(\d+)', parameters)
            expected_n_clusters = m_expected.group(2) if m_expected else "N/A"
        else:
            image_processing = "N/A"
            algorithm = "N/A"
            parameters = "N/A"
            expected_n_clusters = "N/A"
        entry = {
            "File_No": file_id,
            "File_name": model_config,
            "Image_processing": image_processing,
            "Algorithm": algorithm,
            "Parameters": parameters,
            "Expected_n_clusters": expected_n_clusters,
            "Actual_Clusters": res.get("actual_clusters", "N/A"),
            "Silhouette_Score": res.get("silhouette_score", np.nan),
            "Calinski_Harabasz_Score": res.get("calinski_harabasz_score", np.nan),
            "Davies-Bouldin_Index": res.get("davies_bouldin_score", np.nan),
            "WCSS": res.get("wcss", np.nan),
            "Dunn_Index": res.get("dunn_index", np.nan)
        }
        cluster_percentages = res.get("cluster_percentages", {})
        entry.update(cluster_percentages)
        metrics_data.append(entry)
    df_metrics = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(output_dir, "cluster_metrics.xlsx")
    df_metrics.to_excel(metrics_path, index=False)
    print(f"Saved clustering metrics to {metrics_path}")

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """Main function to execute the image clustering evaluation pipeline.

    This function loads clustering assignments from an Excel file, processes each configuration,
    computes various metrics (e.g., silhouette score, WCSS, Dunn Index), visualizes the results using
    multiple dimensionality reduction techniques, and saves all metrics and plots to the output directory.

    Returns:
        None.
    """
    assignments_path = os.path.join("results", "image_clusters.xlsx")
    if not os.path.exists(assignments_path):
        raise ValueError("Clustering assignments file not found. Run first clustering program first.")
    df_assignments = pd.read_excel(assignments_path)
    # Use the list of image names from the Excel file for aggregating results.
    excel_image_names = df_assignments["Image"].tolist()
    results_list = []
    for config in df_assignments.columns:
        if config == "Image":
            continue
        print(f"\nProcessing configuration: {config}")
        tokens = config.split("_")
        if len(tokens) < 2:
            print(f"Configuration '{config}' does not follow expected naming. Skipping.")
            continue
        mod_key = tokens[1]
        mod_folder = os.path.join("data", f"images_{mod_key}")
        if not os.path.exists(mod_folder):
            print(f"Modified image folder '{mod_folder}' not found. Skipping configuration '{config}'.")
            continue
        # Load data through the centralized loader function.
        images, image_names, features = load_data(mod_folder)
        print(f"Feature matrix shape for '{mod_key}': {features.shape}")
        labels = df_assignments[config].to_numpy()
        if np.isnan(labels).all():
            print(f"All cluster assignments are NaN for '{config}'. Skipping visualization...")
            continue
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and len(unique_labels) < len(labels):
            sil_score = silhouette_score(features, labels)
            ch_score = calinski_harabasz_score(features, labels)
            db_score = davies_bouldin_score(features, labels)
            wcss = compute_wcss(mod_folder, labels)
            dunn = compute_dunn_index(mod_folder, labels)
            total = len(labels)
            cluster_percentages = {}
            for cl in np.unique(labels):
                count = np.sum(labels == cl)
                cluster_percentages[f"Cluster{int(cl) + 1}"] = (count / total) * 100
        else:
            sil_score = np.nan
            ch_score = np.nan
            db_score = np.nan
            wcss = np.nan
            dunn = np.nan
            cluster_percentages = {}
        actual_clusters = len(unique_labels)
        print(f"Silhouette Score: {sil_score:.3f}   Calinski–Harabasz Score: {ch_score:.3f}")
        print(f"Davies-Bouldin Index: {db_score:.3f}   WCSS: {wcss:.3f}   Dunn Index: {dunn:.3f}")
        visualize_cluster_images_with_names(mod_folder, labels, config, "results")
        visualize_tsne_config(mod_folder, labels, config, "results")
        visualize_tsne_with_images(mod_folder, labels, config, "results")
        visualize_pca_config(mod_folder, labels, config, "results")
        visualize_pca_with_images(mod_folder, labels, config, "results")
        visualize_projection_config(mod_folder, labels, config, "results")
        results_list.append({
            "model_config": config,
            "algorithm": "N/A",
            "params": {},
            "labels": labels,
            "silhouette_score": sil_score,
            "calinski_harabasz_score": ch_score,
            "davies_bouldin_score": db_score,
            "wcss": wcss,
            "dunn_index": dunn,
            "actual_clusters": actual_clusters,
            "cluster_percentages": cluster_percentages
        })
    save_all_results_excel(results_list, excel_image_names, "results")
    print("Processing completed.")


if __name__ == "__main__":
    main()
