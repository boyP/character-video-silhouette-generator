import tkinter as tk
import cv2
from PIL import Image, ImageTk

# CHANGE ME
video_path = '/Users/pratikprakash/Documents/projects/video-silhouette-generator/input_videos/sue.mov'
video = cv2.VideoCapture(video_path)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# CHANGE ME
output_file = 'sue2.txt'

print(width, height, 'widht, height')

"""

We will keep a running total of current points as new points are drawn for the current frame.
When the user presses next frame, the current points should stay appear and should only be removed when a user actually deletes a point

Keep a dictionary of ids: (x, y) so when we check closest point we can delete that entire hash entry. Then to render the values we cand do dict.values()

"""

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

        print('Num frames:', len(self.frames_list))

        # Add a label above the canvas
        self.label_text = tk.StringVar()
        self.label_text.set(f"Frame {self.current_frame_idx + 1}/{len(self.frames_list)}")
        label = tk.Label(root, textvariable=self.label_text)
        label.pack()

        # Create and set up the canvas to display images
        self.canvas = tk.Canvas(root, width=854, height=480)
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

    def load_frame(self):
        image = self.frames_list[self.current_frame_idx]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((854, 480), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        new_points = {}

        # Draw the points from the previous frame when we load a new frame
        for (x, y) in list(self.current_points.values()):
            red_dot = self.draw_red_dot(x, y, self.dot_radius)
            new_points[red_dot] = (x, y)

        # Reset the existing points with the new set of points we just drew so the ids match
        self.current_points = new_points


    def next_frame(self):
        # Update frame count
        self.current_frame_idx += 1

        print(self.current_frame_idx, 'Frame number')

        # Save the current points into a file
        write_coordinates_to_file(output_file, list(self.current_points.values()))

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