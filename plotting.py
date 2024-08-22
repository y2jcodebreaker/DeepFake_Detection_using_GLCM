import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the CSV files
csv_real_1 = pd.read_csv(r'D:\FaceForensicsLow\FaceForensicsDeepfake\real_faces_csv_1_DeepfakeDetection.csv')
csv_fake_1 = pd.read_csv(r'D:\FaceForensicsLow\FaceForensicsDeepfake\face_2_face_fakefaces_csv_1.csv')

# Extract scalar values from 1D arrays in each feature column
def extract_scalar(arr):
    return np.array(eval(arr))[0]  # Convert the string representation of the array to a list and get the first element

# Process features for both real and fake datasets
features = ['Contrast', 'Dissimilarity', 'Energy', 'ASM', 'Correlation']
for feature in features:
    csv_real_1[feature] = csv_real_1[feature].apply(extract_scalar)
    csv_fake_1[feature] = csv_fake_1[feature].apply(extract_scalar)

    # Convert to numpy arrays
    real_feature = csv_real_1[feature].values
    fake_feature = csv_fake_1[feature].values

    # Print max values
    print(f"Maximum value of Real Faces {feature}: {np.max(real_feature)}")
    print(f"Maximum value of Fake Faces {feature}: {np.max(fake_feature)}")

    # Sort the feature values
    real_feature_sorted = np.sort(real_feature)
    fake_feature_sorted = np.sort(fake_feature)

    # Plot the sorted data
    plt.figure(figsize=(10, 6))
    plt.plot(real_feature_sorted, label=f'Real Faces {feature} (Sorted)')
    plt.plot(fake_feature_sorted, label=f'Fake Faces {feature} (Sorted)')
    
    # Add legend and show plot
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel(feature)
    plt.title(f'Sorted {feature} Comparison: Real vs Fake Faces')
    plt.grid(True)
    plt.show()
