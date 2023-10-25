# video-silhouette-generator

Scripts to generate silhouettes of characters for film. The silhouttes are generated using Meta's [Segment Anything](https://segment-anything.com) model. In order to run these scripts you will need to download the [`sam_vit_h_4b8939.pth`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) model file and add it to the root of this project.

There are 4 scripts in this project

- mps.py
- process_first_frame_silhouette.py
- input_point_drawing_tool/input_point_drawer.py
- input_point_drawing_tool/input_point_drawer_with_output.py

## mps.py

A utility script to identify whether your environment supports Apple metal GPU.

### Usage

```
python3 mps.py
```

## input_point_drawer_with_output.py

A tool with a User interface to draw points on each frame of a video. We display the segmentation mask whenever a new point is drawn. The output is a video of the segmentation masks created by the input points for each frame.

### Usage

```
python input_point_drawing_tool/input_point_drawer_with_output.py
```

## input_point_drawer.py

A tool with a User interface to draw points on each frame of a video. The output is the text file of comma separated list of points drawn on each frame.

For example a text file of a video with two frames would look like this

```
2981,729 3067,1364 2644,1458
2981,729 3067,1364 2644,1458
```

### Usage

```
python3 input_point_drawing_tool/input_point_drawer.py
```

## process_first_frame_silhouette.py

Script to generate segmentation masks using Meta's Segment anything. It takes the video file and the list of input points and outputs a video of the same length segmenting the input from its background. The intention is to run this script after you use `input_point_drawer.py` to generate the list of input points.

### Usage

```
python process_first_frame_silhouette.py
```
