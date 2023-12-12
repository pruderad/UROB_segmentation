import cv2
import os

def extract_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Loop through each frame and save it as an image
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image
        frame_name = f"frame_{i + 1}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Set the path to the video file and the output folder
    video_path = "rb_car.mp4"
    output_folder = "../dataset/unlabeled_dataset/max_car"

    # Extract frames from the video
    extract_frames(video_path, output_folder)