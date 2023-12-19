import torch
import cv2
from utils import *
from fastseg import MobileV3Large, MobileV3Small
from dataloader import UROBDataset
import cv2
import numpy as np
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F


VIDEO_MODE = True


def visualize_segmentation(image, segmentation_mask, unique_labels):

    # Define colors for each class
    colors = {
        0: [0, 0, 0],   # Background (Black)
        1: [255, 0, 0],  # Class 1 (Red)
        2: [0, 0, 255]   # Class 2 (Green)
        # Add more colors for additional classes if needed
    }

    # Create an empty image for overlay
    overlay = image.copy()

    # Overlay segmentation mask on the image
    for label in unique_labels:
        mask = segmentation_mask == label
        color = colors.get(label, [255, 255, 255])  # Default to white for unknown labels
        overlay[mask] = color

    # Blend the overlay with the original image
    alpha = 0.4  # Adjust the transparency of the overlay
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return result

def fix_sizes(img_orig: np.ndarray, target_shape: list):

    #! assuming the labels are resized correctly
    transform = transforms.Compose([transforms.Resize(512, interpolation=Image.BICUBIC, antialias=True)])
    img_pil = Image.fromarray(img_orig)
    transformed_img = np.asarray(transform(img_pil))

    img = np.zeros((*target_shape, 3), dtype=transformed_img.dtype)
    start_idx = (target_shape[1] - transformed_img.shape[1]) // 2 
    assert start_idx >= 0

    img[:, start_idx: start_idx + transformed_img.shape[1], :] = transformed_img

    #print(transformed_img.shape, self.target_shape)
    #print(img.shape, labels.shape)

    return img

def get_all_file_paths(folder_path):
    file_paths = []

    # Walk through all the directories and subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Get the absolute path of each file
            file_path = os.path.abspath(os.path.join(root, file))
            if file_path.endswith('jpg'):
                file_paths.append(file_path)

    return file_paths

# Set the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

# Model and dataset paths
model_path = './saved_models/radim_model01.pt'

unique_labels = [0, 1, 2]

# Load the pre-trained model
#model = MobileV3Small(num_classes=len(unique_labels))
model = MobileV3Large(num_classes=len(unique_labels))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# OpenCV Video Capture from a video file
if VIDEO_MODE:
    video_path = '/home/koondra/Downloads/Race Highlights 2023 Japanese Grand Prix.mp4'  # Provide the path to your video file
    cap = cv2.VideoCapture(video_path)
else:
    filenames = get_all_file_paths('./dataset/tests/')
    filenames.sort()
    index = 0

while True:

    if VIDEO_MODE:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            break

    else:
        if index == len(filenames) - 1:
            break
        frame = np.array(Image.open(filenames[index]))
        index += 1

    # Preprocess the frame
    frame = fix_sizes(frame, [512, 1024])  # Resize to match the model input size
    #frame = np.transpose(frame, (2, 0, 1))
    frame_tensor = torch.tensor(frame / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    frame_tensor = 2* (frame_tensor) - 1 

    # Model prediction
    with torch.no_grad():
        output = model(frame_tensor)
        #print(output.shape)

    # Post-process the segmentation mask
    segmentation_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    probas = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
    #print(segmentation_mask.shape)
    #vis_labels = segmentation_mask[0, :, :]
    print(probas.shape)
    cv2.imshow("depression", np.transpose(probas, (2, 1, 0)))
    #segmented_image = visualize_segmentation(frame, segmentation_mask, unique_labels)

    # Display the result
    #cv2.imshow('Segmentation', segmented_image)
    if not VIDEO_MODE:
        cv2.waitKey(1000)
    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()