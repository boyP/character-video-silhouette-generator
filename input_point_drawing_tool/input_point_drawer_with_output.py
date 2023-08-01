import tkinter as tk
import cv2
from PIL import Image, ImageTk
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# CHANGE ME
video_path = '/Users/pratikprakash/Documents/projects/video-silhouette-generator/input_videos/trimmed_chad_video.mov'
video = cv2.VideoCapture(video_path)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
color_channels = 3
fps = video.get(cv2.CAP_PROP_FPS)

# CHANGE ME
output_file = 'chad2.txt'

print(width, height, 'width, height')

class VideoWriter:
    def __init__(self) -> None:
        # Create VideoWriter to save the output video
        output_path = 'chad2_sam_silhouette_output.mp4' # CHANGE ME
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)

    def write_frame(self, frame) -> None:
        self.output_video.write(frame)


class SegmentAnything:
    def __init__(self) -> None:
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print('Using: ', device)

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(sam)
        sam.to(device=device)

    def load_embeddings(self, frame) -> None:
        # Converts the frame to rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Loads it into sam
        self.predictor.set_image(frame_rgb)

    def _create_mask_frame(self, mask):
        """
        Mask contains the dimensions of the image itself. What values does the mask contain in and out the segmentation. Likely 0
        """
        color = np.array([1, 1, 1]) # White + 100% opacity
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255
        return mask_image

    def get_segmentation_mask(self, input_points_as_list):
        input_points = np.array(input_points_as_list)

        # Computes input_labels from input points
        input_labels = np.ones(len(input_points_as_list), dtype=int)

        # Generate a single mask based on the input points
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        
        # As long as multimask_output is False, there should only be one mask
        assert(len(masks) == 1)
        mask = masks[0] 

        # Generate the black mask over a white background
        frame_with_mask = self._create_mask_frame(mask) # Values between 0 and 1
        frame_with_mask = np.uint8(255 - frame_with_mask)

        return frame_with_mask


def write_coordinates_to_file(filename, coordinates):
    transformed_coordinates = [convert_canvas_to_image_dimensions(x, y) for x, y in coordinates]
    with open(filename, "a") as file:
        coordinates_str = " ".join([f"{x},{y}" for x, y in transformed_coordinates]) + "\n"
        file.write(coordinates_str)

def convert_canvas_to_image_dimensions(x, y):
    return (round(x * width / 854), round(y * height / 480))

class InputPointDrawer:
    def __init__(self, root, video) -> None:
        self.root = root
        self.video = video
        self.frames_list = self.split_video_into_frames()
        self.current_frame_idx = 0
        self.current_points = {}
        self.dot_radius = 5
        self.margin_width = 10

        self.output_video = VideoWriter()

        self.sam = SegmentAnything()

        print('Num frames:', len(self.frames_list))

        # Add a label above the canvas
        self.label_text = tk.StringVar()
        self.label_text.set(f"Frame {self.current_frame_idx + 1}/{len(self.frames_list)}")
        label = tk.Label(root, textvariable=self.label_text)
        label.pack()

        # Create and set up the canvas to display images
        self.canvas = tk.Canvas(root, width=854*2+10, height=480)
        self.canvas.pack()

        # Create the "Next Frame" button
        self.next_frame_button = tk.Button(root, text="Next Frame", command=self.next_frame)
        self.next_frame_button.pack()

        # Bind the click event to the on_canvas_click function
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Bind the right-click event to the delete_red_dot function
        self.canvas.bind("<Button-2>", self.delete_red_dot)

        # Load the first frame
        self.load_frame()

    def show_segmentation_output(self):
        input_points = []
        for (x, y) in self.current_points.values():
            (newX, newY) = convert_canvas_to_image_dimensions(x, y)
            input_points.append([newX, newY])

        self.frame_with_mask = self.sam.get_segmentation_mask(input_points)

        image = cv2.cvtColor(self.frame_with_mask, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((854, 480), Image.LANCZOS)

        # Need to save the photo as a class param so it doesn't get garbage collected
        self.segmented_photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(854 + self.margin_width, 0, image=self.segmented_photo, anchor=tk.NW)

    def load_frame(self):
        og_image = self.frames_list[self.current_frame_idx]
        image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((854, 480), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Load the embeddings for the image
        self.sam.load_embeddings(og_image)
        
        new_points = {}

        # Draw the points from the previous frame when we load a new frame
        for (x, y) in list(self.current_points.values()):
            red_dot = self.draw_red_dot(x, y, self.dot_radius)
            new_points[red_dot] = (x, y)

        # Reset the existing points with the new set of points we just drew so the ids match
        self.current_points = new_points

        # Show a new segementation output if we have existing points on a new frame
        if len(new_points) > 0:
            self.show_segmentation_output()


    def next_frame(self):
        # Update frame count
        self.current_frame_idx += 1

        print(self.current_frame_idx, 'Frame number')

        # Save the current points into a file
        write_coordinates_to_file(output_file, list(self.current_points.values()))

        # Write frame to output video
        self.output_video.write_frame(self.frame_with_mask)

        if self.current_frame_idx >= len(self.frames_list):
            # Close the window
            self.root.destroy()
            return
        
        self.label_text.set(f"Frame {self.current_frame_idx + 1}/{len(self.frames_list)}")

        # Load the next frame
        self.load_frame()

    def split_video_into_frames(self):
        frames = []

        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            frames.append(frame)

        self.video.release()
        return frames
    
    def draw_red_dot(self, x, y, radius):
        return self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="red", outline="red")
    
    def on_canvas_click(self, event):
        dot_x, dot_y = event.x, event.y
        red_dot = self.draw_red_dot(dot_x, dot_y, self.dot_radius)

        # Save in dictionary of ids -> points
        self.current_points[red_dot] = (dot_x, dot_y)

        # Show new segmentation output
        self.show_segmentation_output()

    def delete_red_dot(self, event):
        # Find the closest red dot to the right-clicked position.
        closest_dot = self.canvas.find_closest(event.x, event.y)
        if closest_dot and closest_dot[0] in self.current_points:
            self.canvas.delete(closest_dot[0])
            
            # Need to delete it from current_points too
            del self.current_points[closest_dot[0]]

if __name__ == "__main__":
    root = tk.Tk()
    app = InputPointDrawer(root, video)
    root.mainloop()