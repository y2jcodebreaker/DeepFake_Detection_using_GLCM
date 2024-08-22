import cv2
import os
import numpy as np

# Paths to the input and output directories
input_folder = r'D:\FaceForensicsLow\FaceForensicsDeepfake\fake_frames_Deepfakes'
output_folder = r'D:\FaceForensicsLow\FaceForensicsDeepfake\deepfake_fakefaces'

# Paths to the pre-trained model files
prototxt_path = r'E:\FaceForensicsLow\FaceForensicsDeepfake\deploy.prototxt'
model_path = r'E:\FaceForensicsLow\FaceForensicsDeepfake\res10_300x300_ssd_iter_140000_fp16.caffemodel'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load the pre-trained deep learning model for face detection
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Counter for naming output files
face_counter = 0

# Iterate over all images in the input folder
for filename in os.listdir(input_folder):
    # Build full path to the image file
    image_path = os.path.join(input_folder, filename)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {filename} (not a valid image)")
        continue
    
    (h, w) = image.shape[:2]
    
    # Prepare the image for face detection without resizing
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    # Check for empty detections
    if detections is None or detections.size == 0:
        print(f"No detections for {filename}")
        continue
    
    # Process each detected face
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the box is within image boundaries
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            # Extract the face
            face = image[startY:endY, startX:endX]
            face_filename = f'face{face_counter}.jpg'
            face_path = os.path.join(output_folder, face_filename)
            cv2.imwrite(face_path, face)
            face_counter += 1

print(f"Detected and saved {face_counter} faces.")
