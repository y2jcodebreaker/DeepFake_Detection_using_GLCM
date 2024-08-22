import os

# Specify the folder path
folder_path = r'D:\FaceForensicsLow\FaceForensicsDeepfake\real_faces'

# List all files in the folder
files = os.listdir(folder_path)

# Sort the files (optional)
files.sort()

# Loop through all files and rename them sequentially
for i, filename in enumerate(files):
    # Define new file name with a sequence number
    new_name = f'face{i+1}.png'  # Renames to face1.png, face2.png, etc.

    # Build full file paths
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_name)

    # Rename the file
    os.rename(old_file, new_file)

print("Files renamed successfully!")
