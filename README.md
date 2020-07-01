# Speed tracker

A script to **track objects** and **estimate** their **speed** in videos that move in a straight path perpendicular to the camera.


The script uses OpenCV to track the objects and exports the collected estimated speed data to a csv file.


## Installation

An **OpenCV 3.4+** installation is needed. Either build it from source ([installation guide for Ubuntu](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html)) or install it with
```
pip3 install opencv-contrib-python
```

## Instructions

1. **Run program** with `$ python3 speed_tracker.py video_in csv_out frame_time [-l reference_length] [-r rotation]`


2. Press *r* to **rotate the video** clockwise by 90 degrees. If the video is aligned press *space*.


3. Click and drag to **draw the reference line** with the specified length. By default the reference length is 2 meters. Press *space* to continue.


4. Click and drag to **select the object** you want to track. Press *space* to continue.


Now the tracking starts. The position of the object is cached every frame. 
