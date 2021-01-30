# UberPoC



## Description
UberPoC is an autonomous software able to detect lines on the ground, detect signs, traffic lights and pedestrians.



## Installation

```
 $ git clone https://github.com/PoCFrance/UberPoC
 $ cd UberPoc
```

### Python package dependencies:

- `pyglet`
- `imutils`
- `numpy`
- `matplotlib`
- `opencv-contrib-python`
- `tensorflow`
- `scikit-learn`
- `keras`
- `beautifulsoup4`

## Quick Start
If you want to try Line Detection System through a video or the Duckietown Simulator :
```
$ ./run.py --video-name [video.mp4]
or
$ ./run.py --duckietown
```  
  You can press ``L`` to toggle line detection  
  You can also use ``N`` to see different step of normalization
  
If you want to try Sign Detection over an image or your Camera:
```
$ ./run.py sign_detection --cam
or
$ ./run.py sign_detection --img [img.png]
```
To try Human Detection over an image:
```
$ ./run.py human_detection --img [img.png]
```

## Features

### Implemented

- Line Detection : Displays the lines on the ground while the car is driving.
- Pedestrian Detection: Circle the pedestrians and cyclists on an image.
- Detection and recognition of road signs : Circle and name traffic signs


### Future

- Auto pilot : Combine all the features in order to have an autonomous car
- More road signs and traffic lights : Find dataset of others signs and traffic lights to train AI


## Description of each features

### Line Detection

  - We cut the top of the image in order to gain precision by having only the bottom with the lines
  - Then we apply a red filter on the image to better differentiate the white lines from the rest of the image
  - Using the Canny function of OpenCV, the lines are cut
  - Using the HoughLinesP function we get an array with the different points that make up the lines
  - With these arrays the lines are estimated and displayed on the screen

### Pedestrians Detection

- To detect the pedestrians on a road we use a function of OpenCV (HOGDescriptor_getDefaultPeopleDetector()) who get all the regions of each person on an image
- Then we iterate on these regions and display a rectangle around each pedestrian

### Traffic Signs Detection
- To detect the different traffic signs present on an image or video, we use an AI based on a particular model the "Faster R-CNN" (Region-based Convolutional Neural Networks)
- AI will scan the full image of small regions to be able to detect a sign on the image
- Once the sign is found it will identify it and then surround it on the image with its accuracy rate and type 

## Authors

 - [Thomas Michel](https://github.com/pr0m3th3usEx)
 - [Ugo Levi--Cescutti](https://github.com/ugo94490)
 - [Théo Rocchetti](https://github.com/DCMaker76)
