"""
Script to generate segmentation masks using Meta's Segment anything. It takes the video
file and the list of input points and outputs a video of the same length segmenting the input
from its background.

python process_first_frame_silhouette.py
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

print('Using: ', device)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
sam.to(device=device)

# Load input video CHANGE ME
video_path = '/Users/pratikprakash/Documents/projects/video-silhouette-generator/input_videos/sue.mov'
video = cv2.VideoCapture(video_path)

# CHANGE ME
input_points_file = 'sue2.txt'

# Get video dimensions
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
color_channels = 3
fps = video.get(cv2.CAP_PROP_FPS)

white_background = np.ones((height, width, color_channels), dtype=np.uint8) * 255

# Create VideoWriter to save the output video
output_path = 'sue_sam_silhouette_output.mp4' # CHANGE ME
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)

def create_mask_frame(mask):
    """
    Mask contains the dimensions of the image itself. What values does the mask contain in and out the segmentation. Likely 0
    """
    color = np.array([1, 1, 1]) # White + 100% opacity
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255
    return mask_image

def split_coordinate(coordinate):
    x, y = map(float, coordinate.split(','))
    return [x, y]


def load_coordinates_from_file():
    coordinates = []
    with open(input_points_file, 'r') as file:
        for line in file:
            points = map(split_coordinate, line.strip().split())
            coordinates.append(list(points))
    return coordinates

frame_num = 1

all_input_points = load_coordinates_from_file()

while True:
    print(f"Processing frame #{frame_num}")
    # Process the first frame
    ret, frame = video.read()
    if ret:
    
        # Pull the first video frame and convert it to rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prepare the embeddings
        predictor.set_image(frame_rgb)

        # Get input points. Hardcoded for now
        input_points = np.array(all_input_points[frame_num - 1])
        input_labels = np.ones(len(input_points), dtype=int)

        # Generate a single mask based on the input points
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )

        # As long as multimask_output is False, there should only be one mask
        assert(len(masks) == 1)
        mask = masks[0] 

        # Generate the black mask over a white background
        frame_with_mask = create_mask_frame(mask) # Values between 0 and 1
        frame_with_mask = np.uint8(255 - frame_with_mask)

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2)
        
        # Display the first image in the left subplot
        axes[0].imshow(frame)
        axes[0].axis('off')  # Turn off axis labels
        axes[0].scatter([point[0] for point in input_points], [point[1] for point in input_points], color='red')

        axes[1].imshow(frame_with_mask)
        axes[1].axis('on')  # Turn off axis labels

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.1)
        plt.show()

        # Write the frame to a video
        output_video.write(frame_with_mask)

        frame_num += 1
    else:
        break
        
print("Done")

# Release the video capture
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
