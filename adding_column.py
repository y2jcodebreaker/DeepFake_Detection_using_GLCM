# import pandas as pd

# # Load the CSV file into a DataFrame
# df = pd.read_csv(r'D:\FaceForensicsLow\FaceForensicsDeepfake\real_faces_csv_10.csv')

# # Add the 'Deepfake' column with all values set to 1
# df['Deepfake'] = 0

# # Save the updated DataFrame back to a CSV file
# df.to_csv('real_faces_csv_10_addedColumn.csv', index=False)

import pandas as pd

# Load the first CSV file into a DataFrame
df1 = pd.read_csv(r'D:\FaceForensicsLow\FaceForensicsDeepfake\real_faces_csv_1_addedColumn.csv')

# Load the second CSV file into another DataFrame
df2 = pd.read_csv(r'D:\FaceForensicsLow\FaceForensicsDeepfake\fake_faces_csv_1_addedColumn.csv')

# Concatenate the contents of df2 under df1
df_combined = pd.concat([df1, df2], ignore_index=True)

# Randomize the rows of the combined DataFrame
df_randomized = df_combined.sample(frac=1).reset_index(drop=True)

# Save the randomized DataFrame to a new CSV file (or overwrite one of the original files)
df_randomized.to_csv('faces_combined_randomized_1.csv', index=False)


