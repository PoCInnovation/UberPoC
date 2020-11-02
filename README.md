# UberPoC



## Description
UberPoC is an autonomous car able to follow lines on the ground, detect signs, traffic lights and pedestrians.



## Installation

```
 $ git clone https://github.com/PoCFrance/UberPoC
 $ cd UberPoc
```


## Quick Start
If you want to try Line Detection System through a video or the Duckietown Simulator :
```
$ ./app.py --video-name [video.mp4]
or
$ ./app.py --duckietown
```
If you want to try Sign Detection over an image or your Camera:
```
$ ./app.py sign_detection --cam
or
$ ./app.py sign_detection --img [img.png]
```



## Features



### Implemented

- Line Detection : Displays the lines on the ground while the car is driving.
- Pedestrian Detection: Circle the pedestrians and cyclists on an image.
- Detection and recognition of road signs : Circle and name traffic signs


### Future

- Auto pilot : Combine all the features in order to have an autonomous car
- More road signs and traffic lights : Find dataset of others signs and traffic lights to train AI



## Authors

 - [Thomas Michel](https://github.com/pr0m3th3usEx)
 - [Ugo Levi--Cescutti](https://github.com/ugo94490)
 - [Théo Rocchetti](https://github.com/DCMaker76)
