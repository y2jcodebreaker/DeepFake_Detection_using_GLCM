import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops

# Function to extract and save texture features
def extract_features_and_save(folder_path, output_prefix):
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]  # Adjust extensions as needed

    # Distances to calculate features
    distances = [1, 5, 10]

    # Initialize dictionaries to store features for each distance
    features_dict = {distance: {'Contrast': [], 'Dissimilarity': [], 'Energy': [], 'ASM': [], 'Correlation': []} for distance in distances}

    # Loop through all files in the folder
    for filename in files:
        # Build the full file path
        image_path = os.path.join(folder_path, filename)

        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Resize the image to 128x128
        resized_image = cv2.resize(image, (128, 128))

        # Convert the resized image to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Loop through each distance
        for distance in distances:
            # Calculate the gray-level co-occurrence matrix (GLCM) for the current distance
            g = graycomatrix(gray_image, [distance], [0, np.pi/4, np.pi/2, 3*(np.pi/4)], symmetric=True, normed=True)

            # Compute texture features from the GLCM and flatten the arrays
            contrast = np.expand_dims(graycoprops(g, 'contrast').flatten().mean(), axis=0)
            dissimilarity = np.expand_dims(graycoprops(g, 'dissimilarity').flatten().mean(), axis=0)
            energy = np.expand_dims(graycoprops(g, 'energy').flatten().mean(), axis=0)
            asm = np.expand_dims(graycoprops(g, 'ASM').flatten().mean(), axis=0)
            correlation = np.expand_dims(graycoprops(g, 'correlation').flatten().mean(), axis=0)

            # Append features to the respective lists
            features_dict[distance]['Contrast'].append(contrast)
            features_dict[distance]['Dissimilarity'].append(dissimilarity)
            features_dict[distance]['Energy'].append(energy)
            features_dict[distance]['ASM'].append(asm)
            features_dict[distance]['Correlation'].append(correlation)

    # Create a DataFrame for each distance
    dataframes = {}
    for distance in distances:
        dataframes[distance] = pd.DataFrame(features_dict[distance])
        # Print the shape of the 'Contrast' column
        print(f"Dimensions of 'Contrast' for distance {distance}: {dataframes[distance]['Contrast'].shape}")
        # Save each DataFrame to a CSV file
        # Replace backslashes with forward slashes outside of the f-string
        output_prefix_clean = output_prefix.replace('\\', '/')
        csv_path = f"{output_prefix_clean}_csv_{distance}.csv"
        dataframes[distance].to_csv(csv_path, index=False)
        print(f"Saved DataFrame for distance {distance} to {csv_path}")



    # Optionally, print the DataFrames
    for distance, df in dataframes.items():
        print(f"Distance {distance}:")
        print(df)

# Specify the folder paths for real and fake faces
real_faces_folder = r'D:\FaceForensicsLow\FaceForensicsDeepfake\real_faces'
fake_faces_folder = r'D:\FaceForensicsLow\FaceForensicsDeepfake\face_2_face_fakefaces'

# Extract and save features for real faces
#extract_features_and_save(real_faces_folder, 'D:/FaceForensicsLow/FaceForensicsDeepfake/real_faces')

# Extract and save features for fake faces
extract_features_and_save(fake_faces_folder, r'D:\FaceForensicsLow\FaceForensicsDeepfake\face_2_face_fakefaces')
