# Advanced Lane Finding

The goals of this project are to

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



[//]: # (Image References)

[im01]: figures/Selection_001.jpg "Chessboard Calibration"
[im02]: figures/Selection_002.jpg "Undistorted Chessboard"
[im03]: figures/Selection_003.jpg "Undistorted Dashcam Image"
[im04]: figures/Selection_004.jpg "perspective_transform Dashcam Image"
[im05]: figures/Selection_005.jpg "RGB"
[im06]: figures/Selection_006.jpg "HSV"
[im07]: figures/Selection_007.jpg "HLS"
[im08]: figures/Selection_008.jpg "sobel"
[im09]: figures/Selection_009.jpg "mag_threshold"
[im10]: figures/Selection_010.jpg "direction thersholding"
[im11]: figures/Selection_011.jpg "combination"
[im12]: figures/Selection_012.jpg "HLS l"
[im13]: figures/Selection_013.jpg "lab b"
[im14]: figures/Selection_014.jpg "rect"
[im15]: figures/Selection_015.jpg "Data Drawn onto Original Image"
[im16]: figures/Selection_016.jpg "histogram"
[im17]: figures/Selection_017.jpg "alternalte"
[im18]: figures/Selection_018.jpg "final"

## Camera calibration

The code for this section is in the 2nd and 3rd cells of the ipython notebook`Advanced Lane Detection.ipynb`. The first cell
finds the corners of 20 chessboard images given as examples using the OpenCV function `findChessboardCorners`. This function finds the objpoints and image points. These are fed into the `cv2.undistort` to get the distortion matrix used to undistort the image.
Here is the detected chessboard corners
![alt text][im01]

Here is the undistorted image
![alt text][im02]



## Experimenting with single images

This distortion matrix is used to undistort the lane images cell number 5 shows the sample undistorted image.
![alt text][im03]

I then applied the perspective transform and defined a function called `wrap`. This function takes in image, src(source) and dst(destination) matrices. I have manually coded these matrices in the cell #7 in the notebook. After applying perspective transform we have the following image
![alt text][im04]

### Color channels
I then explored individual RGB channels to detect the best channel for detecting lane line
![alt text][im05]

I then explored HSV channels
![alt text][im06]

I then explored the HLS channels
![alt text][im07]

### Gradient thresholding
I then applied gradient thresholding to orginal images. I used all the three different techiniques taught the course

1) absoulte x and y gradient using `cv2.Sobel` function the corresponding code is found in the cells 13 and 14
![alt text][im08]

2) Magnitude thresholding here i used the kernal size as 25 and min thresh =25 and max thresh = 255
![alt text][im09]

3) Direction thresholding. The corresponding function is in the cells 17 and 18 where i used the kernal size = 7 and min threshold is 0.0 and max thershold is 0.09
![alt text][im10]

I then combined all the 3 techniques and found the best using different thresholding
![alt text][im11]

### Color channels continued
I first used the HLS channels and used thresholds to detect lane lines. I found out the L channel is best for while lane; and thershold (220,255) are used in the custom function `hls_binary`.
![alt text][im12]

and then used the B channel in the Lab color space for detecting the yellow lines. Found in the cell 25.
![alt text][im13]

I then used the above developed structure and then used it on all the test images given the output is shown below
![alt text][im14]

## Polyfit and histograms

I then used the given framework in the course to fit a polynomial to the lines detected using the above structure and also used the sliding window to detect the lane lines along throgh out the image
![alt text][im15]

the corresponding histogram is shown below
![alt text][im16]

I have also used an alternate method used to make continuas polyfit using the same code used in the course
![alt text][im17]

## Curvature and center
I have then used the function curv_rad function to determine the radius of curvature and center of the car in the lane. Using all the framework generated we have the following image
![alt text][im18]

I have then used the moviepy package to apply to the project video the code was successfully able to detect all the lane lines throughout the video. The link for the video is [here](https://github.com/sai19872000/Advanced_road_lane_detection/blob/master/project_video_output.mp4)


## Discussion 
This project was pretty straight forward. The main challenge was to detect the yellow lines. After some online researching I found out that the Lab (B) channel is the best for this purpose. My code had some trouble applying to challenge video, more fine tuning is necessary for this purposes.
