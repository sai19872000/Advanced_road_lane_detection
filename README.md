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

#read all the calibration images 
import glob
matplotlib qt

images = glob.glob('calibration*.jpg')

## reading the image
img = mpimg.read('calibration_image1.png')
plt.imshow(img)

objpoints = []
imgpoints = []

objp = np.zeros((8*6,3),np.float32)
objp[:,:2] = np.mgrid(0:8,0:6).T.reshape(-1,2)

for fname in images:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray,(8,6),None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        img = cv2.drawChessboardCorners(img,(8,6),corners,ret)
        plt.imshow(img)
 ```
 
 The two important cv2 functions 
 ```python
 ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 ```
 
 used to get thr cooefficients of distrortion "dist"
 camera matrix "mtx"
 "rvecs and tvecs" for getting the position on the real world
 another function is the funtion to undistort the image; this funtion takes in the image and camera matrix and distortion coefficient 
 
 ```python
 dst = cv2.undistort(img, mtx, dist, None, mtx)
 ```
 ##Quiz 2 (undistoring images)

```python 
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

# Read in an image
img = cv2.imread('test_image.png')

# TODO: Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #undist = np.copy(img)  # Delete this line
    return undist

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
undistorted = cal_undistort(img, objpoints, imgpoints)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```

step1 edge detection and masking 
step2 perspective transform
step3 get the curvature using a polynomial fit (2nd order polynomial)

f(y) = Ay^2 + By + C, where A, B, and C are coefficients.

A gives you the curvature of the lane line, B gives you the heading or direction that the line is pointing, and C gives you the position of the line based on how far away it is from the very left of an image (y = 0).

## Persective transfrom using openCV

Perspective transform

A perspective transform maps the points in a given image to different, desired, image points with a new perspective. The perspective transform you’ll be most interested in is a bird’s-eye view transform that let’s us view a lane from above; this will be useful for calculating the lane curvature later on. Aside from creating a bird’s eye view representation of an image, a perspective transform can also be used for all kinds of different view points.

python implementation 

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib qt

img = mpimg.imread('../road_images_video/stopsign.jpg')

plt.imshow(img)

%matplotllib inline
plt.imshow(img)
plt.plot(850,320,'.')
plt.plot(865,450,'.')
plt.plot(533,350,'.')
plt.plot(535,210,'.')

```
