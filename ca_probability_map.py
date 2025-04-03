import os
import locale
import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
import seaborn as sns


def load_image_as_array(img_path: str) -> np.ndarray:
    """Load an image file and return a grayscale numpy array.

    Args:
        img_path (str): Path to the image file.

    Returns:
        np.ndarray: Grayscale image as a numpy array.
    """
    with Image.open(img_path).convert('L') as im:
        return np.array(im, dtype=np.uint8)


def _compute_probability_maps_data(strategy: str,
                                   images_folder: str,
                                   clusters_file: str) -> (dict, np.ndarray):
    """Compute probability maps for each cluster and the global probability map.

    This helper function reads an Excel file with image-to-cluster assignments,
    groups images by cluster based on a specified strategy, and computes the
    per-pixel probability of calcification from binary images (with values 0 or 255).
    The probability is computed by summing pixel values and normalizing by (255 * number_of_images).

    Args:
        strategy (str): Identifier substring for the clustering strategy (e.g., '0346').
        images_folder (str): Path to the folder containing thresholded binary images.
        clusters_file (str): Path to the Excel file with image-to-cluster assignments.

    Returns:
        tuple: A tuple containing:
            - cluster_probability_maps (dict): Mapping from cluster label (str) to its probability map (np.ndarray).
            - global_probability_map (np.ndarray): The global probability map computed from all images.

    Raises:
        ValueError: If no column matching the strategy is found or if no images are available.
    """
    # Read Excel file and identify the column matching the strategy.
    df = pd.read_excel(clusters_file)
    filename_column = df.columns[0]
    matching_cols = [col for col in df.columns if strategy in col]
    if not matching_cols:
        raise ValueError(
            f"No column containing '{strategy}' found in '{clusters_file}'. "
            f"Available columns: {df.columns.tolist()}"
        )
    chosen_col = matching_cols[0]

    # Filter rows with valid cluster labels.
    df_strategy = df[[filename_column, chosen_col]].dropna(subset=[chosen_col])

    # Group images by cluster label.
    clusters_dict = {}
    for _, row in df_strategy.iterrows():
        image_name = str(row[filename_column])
        cluster_label = str(row[chosen_col])
        clusters_dict.setdefault(cluster_label, []).append(image_name)

    # Global list of images.
    all_images = df_strategy[filename_column].unique().tolist()
    if len(all_images) == 0:
        raise ValueError(f"No images found for strategy '{strategy}'.")

    # Load a sample image to determine shape.
    example_img_path = os.path.join(images_folder, all_images[0])
    example_img = load_image_as_array(example_img_path)
    height, width = example_img.shape

    # Initialize accumulators.
    global_sum_array = np.zeros((height, width), dtype=np.float64)
    global_count = 0
    cluster_sums = {}
    cluster_counts = {}

    for c_label, img_list in clusters_dict.items():
        cluster_sums[c_label] = np.zeros((height, width), dtype=np.float64)
        cluster_counts[c_label] = len(img_list)

    # Accumulate pixel values per cluster and globally.
    for c_label, img_list in clusters_dict.items():
        for img_name in img_list:
            img_path = os.path.join(images_folder, img_name)
            if not os.path.isfile(img_path):
                print(f"Warning: file '{img_path}' not found. Skipping.")
                continue
            img_array = load_image_as_array(img_path)
            cluster_sums[c_label] += img_array
            global_sum_array += img_array
            global_count += 1

    # Compute probability maps (values normalized between 0 and 1).
    cluster_probability_maps = {}
    for c_label, sum_array in cluster_sums.items():
        count = cluster_counts[c_label]
        if count > 0:
            cluster_probability_maps[c_label] = sum_array / (255.0 * count)
        else:
            cluster_probability_maps[c_label] = np.zeros((height, width), dtype=np.float64)

    if global_count > 0:
        global_probability_map = global_sum_array / (255.0 * global_count)
    else:
        global_probability_map = np.zeros((height, width), dtype=np.float64)

    return cluster_probability_maps, global_probability_map


def probability_map_full(cluster_probability_maps: dict,
                         global_probability_map: np.ndarray,
                         strategy: str,
                         output_folder: str) -> None:
    """Plot and export the full probability maps figure.

    This function creates a figure with one subplot per cluster and one subplot for the global map.
    Each probability map is displayed with a 'jet' colormap and a shared colorbar using comma
    as the decimal separator. The resulting figure is saved in the output folder.

    Args:
        cluster_probability_maps (dict): Mapping from cluster labels to probability maps (np.ndarray).
        global_probability_map (np.ndarray): Global probability map computed from all images.
        strategy (str): Identifier for the clustering strategy.
        output_folder (str): Folder where the full probability map figure will be saved.

    Returns:
        None
    """
    # Ensure output folder exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set font properties.
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 14

    cluster_labels_sorted = sorted(cluster_probability_maps.keys())
    num_clusters = len(cluster_labels_sorted)

    # Create a subplot for each cluster plus one for the global map.
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_clusters + 1,
        figsize=(5 * (num_clusters + 1), 5),
        constrained_layout=True
    )
    if num_clusters == 1:
        axes = [axes]

    im = None
    for i, c_label in enumerate(cluster_labels_sorted):
        ax = axes[i]
        prob_map = cluster_probability_maps[c_label]
        im = ax.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
        ax.set_title(f"Cluster {c_label}\n(strategy {strategy})")

    # Global map in the last subplot.
    ax_global = axes[-1] if num_clusters > 1 else axes[0]
    im = ax_global.imshow(global_probability_map, cmap='jet', vmin=0, vmax=1)
    ax_global.set_title(f"Global Map\n(strategy {strategy})")

    # Formatter for the colorbar.
    def comma_format(x, pos):
        return f"{x:.2f}".replace('.', ',')

    cbar = fig.colorbar(
        im,
        ax=axes if num_clusters > 1 else axes[0],
        fraction=0.02,
        pad=0.04
    )
    cbar.set_label("Вероятность кальцификации", rotation=270, labelpad=15)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(comma_format))
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    output_figure_path = os.path.join(output_folder, f"probmaps_strategy_{strategy}.png")
    plt.savefig(output_figure_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Full probability map saved to '{output_figure_path}'.")


def probability_tresholded_map(cluster_probability_maps: dict,
                               strategy: str,
                               images_raw_folder: str,
                               output_folder: str,
                               sample_fraction: float = 0.5) -> None:
    """Export thresholded probability maps as colored composite images with sampled background.

    For each cluster, this function groups raw images from the clustering file and samples a
    fraction of them to build a background mask. The background mask is the union of nonzero
    pixels from the sampled raw images. Then, for thresholds ranging from 5% to 100% (in increments
    of 5%), a composite image is created. In the composite image, for each pixel in the background mask,
    if the probability is equal to or above the threshold, the pixel is set to the threshold color
    (the rightmost color of the seaborn palette); otherwise, it is set to the background color
    (the leftmost color of the palette). Pixels outside the background mask remain black.

    Args:
        cluster_probability_maps (dict): Mapping from cluster labels to probability maps (np.ndarray).
        strategy (str): Identifier for the clustering strategy.
        images_raw_folder (str): Folder containing the original raw images.
        output_folder (str): Folder where the thresholded composite images will be saved.
        sample_fraction (float, optional): Fraction of raw images to sample for each cluster. Defaults to 0.5.

    Returns:
        None
    """
    clusters_file = 'results/image_clusters.xlsx'

    # Read clustering file to group raw images by cluster.
    df = pd.read_excel(clusters_file)
    filename_column = df.columns[0]
    matching_cols = [col for col in df.columns if strategy in col]
    if not matching_cols:
        raise ValueError(
            f"No column containing '{strategy}' found in '{clusters_file}'. "
            f"Available columns: {df.columns.tolist()}"
        )
    chosen_col = matching_cols[0]
    df_strategy = df[[filename_column, chosen_col]].dropna(subset=[chosen_col])

    # Build dictionary mapping each cluster to a list of raw image filenames.
    cluster_to_images = {}
    for _, row in df_strategy.iterrows():
        c_label = str(row[chosen_col])
        cluster_to_images.setdefault(c_label, []).append(str(row[filename_column]))

    method_output_dir = os.path.join(output_folder, strategy)
    os.makedirs(method_output_dir, exist_ok=True)

    # Obtain colors from the seaborn palette.
    palette = sns.color_palette("plasma")
    background_color = tuple(int(255 * x) for x in palette[0])
    threshold_color = tuple(int(255 * x) for x in palette[-1])

    # Iterate over clusters.
    for c_label, prob_map in cluster_probability_maps.items():
        # Get list of raw images for this cluster.
        raw_images_list = cluster_to_images.get(c_label, [])
        if not raw_images_list:
            print(f"Warning: No raw images found for cluster '{c_label}'. Skipping.")
            continue

        # Determine number of images to sample.
        num_to_sample = max(1, int(len(raw_images_list) * sample_fraction))
        sampled_images = raw_images_list

        # Initialize background mask as None.
        background_mask = None

        # Compute the union mask (logical OR) of nonzero pixels from sampled raw images.
        for img_name in sampled_images:
            raw_img_path = os.path.join(images_raw_folder, img_name)
            if not os.path.isfile(raw_img_path):
                print(f"Warning: Raw image '{raw_img_path}' not found for cluster '{c_label}'. Skipping image.")
                continue
            raw_img_array = load_image_as_array(raw_img_path)
            mask = raw_img_array > 0
            if background_mask is None:
                background_mask = mask
            else:
                background_mask |= mask

        if background_mask is None:
            print(f"Warning: No valid raw images for cluster '{c_label}'. Skipping cluster.")
            continue

        height, width = background_mask.shape

        # For each threshold, create and save the composite image.
        for threshold in np.arange(0.05, 1.05, 0.05):
            composite_img = np.zeros((height, width, 3), dtype=np.uint8)
            # Compute threshold masks.
            mask_thresh = (prob_map >= threshold) & background_mask
            mask_bg = background_mask & ~(prob_map >= threshold)
            composite_img[mask_thresh] = threshold_color
            composite_img[mask_bg] = background_color
            # Pixels outside background_mask remain black.

            threshold_percent = int(round(threshold * 100))
            file_name = f"cluster_{c_label}_threshold_{threshold_percent:02d}.png"
            file_path = os.path.join(method_output_dir, file_name)
            Image.fromarray(composite_img, mode='RGB').save(file_path)

    print(f"Thresholded composite maps saved in '{method_output_dir}'.")


def main() -> None:
    """Main function to compute and export probability maps.

    This function defines the parameters, computes the per-cluster and global probability maps,
    exports the full probability map figure, and then exports colored thresholded composite images.
    It serves as the main entry point of the program.

    Returns:
        None
    """
    # Define parameters.
    strategy = '0346'
    images_threshold_folder = 'data/images_threshold'
    images_raw_folder = 'data/images_raw'
    clusters_file = 'results/image_clusters.xlsx'
    output_folder = 'results/probability_maps'

    # (Optional) set locale for decimal commas if needed.
    # locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')

    # Compute probability maps data.
    cluster_prob_maps, global_prob_map = _compute_probability_maps_data(
        strategy, images_threshold_folder, clusters_file
    )

    # Export full probability map figure.
    probability_map_full(cluster_prob_maps, global_prob_map, strategy, output_folder)

    # Export colored thresholded composite maps with sampled raw images.
    probability_tresholded_map(cluster_prob_maps, strategy, images_raw_folder, output_folder)


if __name__ == "__main__":
    main()
