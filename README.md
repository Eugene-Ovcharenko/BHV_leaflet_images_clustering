# Image Clustering, Visualization, and Statistical Analysis Pipeline v1.5

This pipeline comprises three Python scripts that provide a comprehensive workflow for preprocessing, clustering, visualization, and statistical analysis of Multislice Computed Tomography (MSCT) data of scanned prosthetic leaflets explanted due to Structural Valve Degeneration (SVD). The scripts incorporate additional functionality for calculating image-level metrics, as well as advanced probability mapping that generates colored composite maps. Enhanced visualization functions display thumbnail images (with optional annotations) on t‑SNE and PCA scatter plots, in addition to grid views that group images by clusters.

---

## 1. `image_clustering.py`

- **Input:**  
  Grayscale PNG images from `data/images`.

- **Processing:**  
  - **Image Loading and Preprocessing:**  
    Loads images, converts them to grayscale (if necessary), and resizes them.
  - **Image Modification Strategies:**  
    Applies three strategies:
    - **raw:** The original image (no modifications).
    - **threshold:** Fixed thresholding—pixels with values in [193, 255] are set to 255 (calcium), while all other pixels are set to 0.
    - **combined:** Each image is processed so that:
      - Pixels in [193, 255] become 255 (calcium),
      - Pixels in [1, 192] become 124 (biological tissue),
      - Pixels with value 0 remain unchanged.
  - **Clustering:**  
    Clusters the images using multiple algorithms (KMeans, Gaussian Mixture Model, Spectral Clustering, Agglomerative Clustering, OPTICS) with hyperparameter optimization.

- **Output:**  
  - Modified images are saved in subdirectories (e.g., `data/images_raw`, `data/images_threshold`, `data/images_combined`).
  - Clustering assignments are stored in `results/image_clusters.xlsx`.

---

## 2. `clusters_analysis.py`

- **Input:**  
  - Clustering assignments from `results/image_clusters.xlsx`.  
  - Clinical data from `data/data_clinical.xlsx`.

- **Processing:**  
  - **Clustering Metrics:**  
    Computes clustering quality metrics including:
    - Silhouette Score
    - Calinski–Harabasz Score
    - Davies–Bouldin Index
    - Dunn Index
    - WCSS
  - **New Functionality:**  
    - **Image Metrics Calculation:**  
      Calculates per-image metrics:
      - Count of non-zero pixels.
      - Count of white pixels (pixel value 255).
      - Ratio of white to non-zero pixels (formatted as a percentage).
    - **Patient Name Extraction:**  
      Extracts an identifier from each image filename (assumed format: `ID_010_3.png`) and retrieves the corresponding patient name from the clinical data.
    - Saves the computed metrics and patient names in an Excel file named `results/calcium_percentage.xlsx`.
  - **Visualization:**  
    Generates a variety of visualizations following GOST-style guidelines:
    - **t-SNE Scatter Plots:** Standard scatter plots with markers.
    - **t-SNE with Images:** t-SNE plots where thumbnail images replace markers.
    - **PCA Scatter Plots:** Standard PCA scatter plots with markers.
    - **PCA with Images:** PCA plots displaying thumbnail images instead of markers.
    - **Cluster Image Grids:** Grid views grouping images by cluster with image names shown below each image.
    - **Projection Plots:** Maximum Intensity Projection (MIP) plots per cluster.
  - Saves all generated plots as PNG files and updates overall metrics in `results/cluster_metrics.xlsx`.

- **Output:**  
  Visualizations and metrics are saved in the `results` directory, including the new `calcium_percentage.xlsx` file.

---

## 3. `ca_probability_map.py`

- **Input:**  
  - Thresholded (binary) images from `data/images_threshold` (with pixel values 0/255).  
  - Clustering assignments from `results/image_clusters.xlsx`.

- **Processing:**  
  - Reads the Excel file and identifies the column whose name contains the provided strategy substring (e.g., `0346`).
  - Groups images by their cluster labels from that column.
  - For each cluster:
    - Sums pixel values across all images and computes the per-pixel probability of calcification.
    - Generates a composite colored probability map where:
      - The background is computed from the union of all nonzero pixels across raw images in each cluster.
      - For thresholds ranging from 5% to 100% (in 5% increments), pixels meeting or exceeding the threshold are highlighted with a designated threshold color (from a seaborn palette), while the remaining background pixels are colored with the background color.
  - Produces a single figure with subplots showing each cluster’s probability map and an additional subplot for the global map (all images combined).

- **Output:**  
  A single figure (PNG) with colored probability maps is saved in `results/probability_maps/` under the name `probmaps_strategy_{strategy}.png`.

---

## Usage

1. **`python image_clustering.py`**  
   Preprocesses, modifies, and clusters images; outputs `results/image_clusters.xlsx`.

2. **`python clusters_analysis.py`**  
   Computes clustering metrics, calculates image-level metrics, extracts patient names, and generates visualizations; outputs `results/cluster_metrics.xlsx` and `results/calcium_percentage.xlsx`.

3. **`python ca_probability_map.py --strategy <strategy_substring>`**  
   Generates composite colored probability maps; outputs a figure in `results/probability_maps/`.

---

## Dependencies

- Python 3.7+
- `numpy`, `pandas`, `opencv-python`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `openpyxl`
