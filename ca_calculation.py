import os
import numpy as np
import pandas as pd
from image_clustering import load_images


def calculate_metrics(images):
    """Calculate metrics for each image.

    This function calculates the following metrics for each image:
      - Count of non-zero pixels.
      - Count of white pixels (pixel value 255).
      - Ratio of white pixels to non-zero pixels, formatted as a percentage.

    Args:
        images (list[np.ndarray]): List of image arrays.

    Returns:
        tuple: A tuple containing:
            - non_zero_counts (list[int]): Non-zero pixel counts for each image.
            - white_counts (list[int]): White pixel counts for each image.
            - white_ratios (list[str]): White-to-nonzero pixel ratios as percentage strings.
    """
    non_zero_counts = []
    white_counts = []
    white_ratios = []

    for img in images:
        non_zero = np.count_nonzero(img)
        white = np.sum(img == 255)
        ratio = white / non_zero if non_zero != 0 else 0
        ratio_str = f"{ratio * 100:.2f}%"

        non_zero_counts.append(non_zero)
        white_counts.append(white)
        white_ratios.append(ratio_str)

    return non_zero_counts, white_counts, white_ratios


def extract_id(image_name):
    """Extract the identifier from the image filename.

    It is assumed that the filename follows the format 'ID_010_3.png', where '010' is the identifier.

    Args:
        image_name (str): The image filename.

    Returns:
        str or None: The extracted identifier or None if it cannot be extracted.
    """
    parts = image_name.split('_')
    if len(parts) >= 2:
        return parts[1]
    return None


def get_patient_names(image_names, clinical_df):
    """Retrieve patient names for each image based on clinical data.

    For each image filename, the function extracts an identifier and searches the clinical data
    for a matching ID to obtain the patient's name.

    Args:
        image_names (list[str]): List of image filenames.
        clinical_df (pd.DataFrame): DataFrame containing clinical data with an 'ID' column and a 'Пациент' column.

    Returns:
        list: A list of patient names corresponding to the image filenames.
    """
    # If the 'ID' column is of type object, strip extra spaces.
    if clinical_df['ID'].dtype == object:
        clinical_df['ID'] = clinical_df['ID'].str.strip()
    else:
        # If IDs are numeric, convert them to int.
        clinical_df['ID'] = clinical_df['ID'].astype(int)

    patient_names = []

    for image_name in image_names:
        id_str = extract_id(image_name)  # e.g., "010"
        try:
            # Convert the extracted string to an integer (e.g., "010" -> 10).
            id_int = int(id_str)
        except (ValueError, TypeError):
            id_int = None

        if id_int is not None:
            # Compare using the integer identifier.
            matching_rows = clinical_df[clinical_df['ID'] == id_int]
            if not matching_rows.empty:
                patient = matching_rows.iloc[0]['Пациент']
            else:
                patient = None
        else:
            patient = None
        patient_names.append(patient)

    return patient_names


if __name__ == "__main__":
    # Load images and their filenames.
    image_path = 'data/images_combined'
    images, image_names = load_images(image_path)

    # Load clinical data.
    clinical_path = 'data/data_clinical.xlsx'
    clinical_df = pd.read_excel(clinical_path)

    # Uncomment the following lines to inspect the clinical data if needed.
    # print(clinical_df.head())
    # print(clinical_df.columns)

    # Calculate metrics for the images.
    non_zero_counts, white_counts, white_ratios = calculate_metrics(images)

    # Retrieve patient names for each image.
    patient_names = get_patient_names(image_names, clinical_df)

    # Create a DataFrame with the required column order.
    df = pd.DataFrame({
        'Image_names': image_names,
        'Пациент': patient_names,
        'Non_Zero_Count': non_zero_counts,
        'White_Count': white_counts,
        'White_Ratio': white_ratios
    })

    # Create the results directory if it does not exist.
    os.makedirs('results', exist_ok=True)

    # Save the DataFrame to an Excel file.
    df.to_excel('results/calcium_percentage.xlsx', index=False)
