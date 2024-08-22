import dlib
import cv2
import os
from PIL import Image

# Set paths
input_image_directory = r'D:\FaceForensicsLow\FaceForensicsDeepfake\face_2_face_fake_frames'
output_faces_directory = r'D:\FaceForensicsLow\FaceForensicsDeepfake\face_2_face_fakefaces'

# Ensure output directory exists
if not os.path.exists(output_faces_directory):
    os.makedirs(output_faces_directory)

# Load the pre-trained face detector from dlib
detector = dlib.get_frontal_face_detector()

def detect_and_save_faces(image_path, output_dir):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert image from BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detector(rgb_image)
    
    # Get image dimensions
    image_height, image_width = rgb_image.shape[:2]
    
    # Process each face
    for i, rect in enumerate(faces):
        x, y, w, h = (rect.left(), rect.top(), rect.right(), rect.bottom())
        
        # Clip the coordinates to ensure they are within the image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(image_width, w)
        h = min(image_height, h)
        
        # Extract face region
        face_image = rgb_image[y:h, x:w]
        
        # Convert to PIL Image and save
        pil_image = Image.fromarray(face_image)
        face_image_path = os.path.join(output_dir, f'{os.path.basename(image_path).split(".")[0]}_face_{i}.png')
        pil_image.save(face_image_path)
        print(f"Saved face {i+1} to {face_image_path}")

def process_images_from_directory(input_dir, output_dir):
    # Process each image file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            detect_and_save_faces(image_path, output_dir)

# Run the face detection and saving process
process_images_from_directory(input_image_directory, output_faces_directory)
