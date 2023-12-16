import cv2
import os
from PIL import Image
from torchvision import transforms
import numpy as np

transform = transforms.Compose([transforms.Resize(512, Image.BICUBIC)])

def extract_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    avail_samples = os.listdir(output_folder)
    max_id = 0
    for filename in avail_samples:
        file_id = int(filename.split('.')[0][6:])
        max_id = max(max_id, file_id)
    print(max_id + 1)

    # Loop through each frame and save it as an image
    for i in range(frame_count):
        ret, frame = cap.read()

        # resize the image
        img_pil = Image.fromarray(frame)
        frame = np.asarray(transform(img_pil))

        if not ret:
            break
        # Save the frame as an image
        if i % 80 == 0:
            frame_name = f"frame_{max_id + 1}.jpg"
            max_id += 1
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)

if __name__ == "__main__":
    # Set the path to the video file and the output folder
    video_path = "rb_car1.mp4"
    output_folder = "../dataset/unlabeled_dataset/max_car"

    # Extract frames from the video
    extract_frames(video_path, output_folder)