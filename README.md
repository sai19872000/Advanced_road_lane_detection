# Advanced_road_lane_detection
Advanced lane finding algorithm openCV

Image Distortion is used to change the perspective of the image

Notes from udacity:
Image distortion occurs when a camera looks at 3D objects in the real world and transforms them into a 2D image;
this transformation isn’t perfect. Distortion actually changes what the shape and size of these 3D objects appear to be. 
So, the first step in analyzing camera images, is to undo this distortion so that you can get correct and useful 
information out of them.


Open Cv chess board corners

Finding Corners
In this exercise, you'll use the OpenCV functions 
```python 
findChessboardCorners()
```and
```python 
drawChessboardCorners() 
```to 
automatically find and draw corners in an image of a chessboard pattern.

sample ipython notebooks

https://github.com/JustinHeaton/Advanced-Lane-Finding
https://github.com/jeremy-shannon/CarND-Advanced-Lane-Lines

Quiz 1

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
nx = 8#TODO: enter the number of inside corners in x
ny = 6#TODO: enter the number of inside corners in y

# Make a list of calibration images
fname = 'calibration_test.png'
img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
```

at least have 20 images for calibration; and one test image
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

## reading the image
img = mpimg.read('calibration_image1.png')
plt.imshow(img)

objectpoints = []
imagepoints = []

objp = np.zeros((8*6,3),np.float32)
objp[:,:2] = np.mgrid(0:8,0:6).T.reshape(-1,2)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray,(8,6),None)
if ret == True:
    img
