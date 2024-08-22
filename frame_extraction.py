import cv2
import os

def extract_frames(video_folder, output_folder, fps=1):
    """
    Extract frames from all videos in the specified folder and save them to the output folder.

    Args:
        video_folder (str): Path to the folder containing videos.
        output_folder (str): Path to the folder where frames will be saved.
        fps (int): Number of frames to extract per second of video.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    frame_count = 11924

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video file {video_file}")
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps // fps)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if current_frame % frame_interval == 0:
                frame_filename = f"frame{frame_count}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                print(f"Saved {frame_path}")
                frame_count += 1

        cap.release()

    print("Frame extraction completed.")

if __name__ == "__main__":
    # Define paths to the folders
    video_folder = r'D:\FaceForensicsLow\manipulated_sequences\Face2Face\c40\videos'
    output_folder = r'D:\FaceForensicsLow\FaceForensicsDeepfake\face_2_face_fake_frames'

    # Extract frames from the videos
    extract_frames(video_folder, output_folder, fps=1)
