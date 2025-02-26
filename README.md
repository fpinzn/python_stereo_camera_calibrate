# What this is
Stereo camera calibration script written in python. Uses OpenCV primarily. 

Experiment 3D landmarks from 2 calibrated cameras 

## Objective

Explore 3d reconstruction using two phones.


## Time Budget:

1 day (it ended up being 9:30h)
Saimon was present for 3 hours and his help was useful in marking where the view of the camera started/ended, Francisco finished the experiment by himself.

## 25/02/2025 

We followed this guys advice: 

https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html

Cloned this repo:

https://github.com/TemugeB/python_stereo_camera_calibrate?tab=readme-ov-file

First we tried using a perpendicular setup for the cameras but it wasnt possible to get OpenCV to recognize the checkerboard pattern in frames of both cameras at the same time

In a ~45deg angle from one camera to the like this:

￼

We could get this couple frames:

￼

We adapted the repo using ChatGPT to calibrate the intrinsic and extrinsic params of both cameras from videos instead of live streams of connected devices. To be able to do this both videos were edited in premier to be synched and the same length.

calib.py receives the videos in this order:  

`python calib.py calibration_settings.yaml videos/03-tobias-front_1.mp4 videos/03-tobias-side_1.mp4`

That returns the calibration results in console like this:
```
--- Calibration Parameters ---
Camera0 Intrinsic Matrix:
 [[229.75032642   0.         200.44588524]
 [  0.         169.79368238 246.51861282]
 [  0.           0.           1.        ]]
Camera0 Distortion Coefficients:
 [[-0.14493007  0.03117784  0.09789049 -0.290851    0.17057335]]
Camera1 Intrinsic Matrix:
 [[185.66584519   0.         206.46116634]
 [  0.         159.65919947 216.49924977]
 [  0.           0.           1.        ]]
Camera1 Distortion Coefficients:
 [[-0.04283384  0.04385798 -0.05661369  0.03800788 -0.02296988]]
Stereo Rotation Matrix (Camera0 -> Camera1):
 [[ 0.07658793 -0.24328972  0.96692523]
 [-0.42846919  0.86763286  0.25224466]
 [-0.90030464 -0.43361656 -0.03779194]]
Stereo Translation Vector (Camera0 -> Camera1):
 [[-124.13846774]
 [ -20.27274023]
 [  14.12344011]]
------------------------------
```
From memory, the RMSE (Root Mean Square Error) was between 3.something and 4.something. Which according to the dudes blogpost is way too high.

Using that and Chat GPT the extract_video_landmarks.sh file was created, it creates a swift script called ‘pose-estimation-video.swift’ which compiles and executes to create front_video.json and side_video.json, the last step is to join both files in a csv called video_landmarks.csv

to run this do:

`./extract_video_landmarks.sh videos/03-tobias-front_1.mp4 videos/03-tobias-side_1.mp4`


And then to display the landmarks do

`python triangulate.py video_landmarks.csv`

￼

### Results

It seems like a promising approach. The “front” camera was too far from the dog and thats probably the reason why the limbs are missing in most of the frames. Also, maybe the positioning of the cameras could be a little closer AND the checkerboard pattern a little bigger even if that means less rows and columns.

Some more theoretical background could be immensely useful, as my intuition was a that a perpendicular positioning would be better but in several experiments a more “side to side” approach has been observed.


### Next Steps:

1. Read a bit more on how the technique works and how cameras are usually set up for it and follow that
2. Organize the direction of the displayed 3d model (maybe lock the graph with the correct direction for each dimension) so its easier to understand it


