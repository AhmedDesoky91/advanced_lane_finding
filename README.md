# **Advanced Lane Lines Finding Using Python and OpenCV**
---
![intro](https://i.imgur.com/dZMacfB.jpg)

**In this project, I have detected lane lines on the road in several circumstances, straight lane lines, curvy, and in different lighting conditions using _Python_ and _OpenCV_.**

**Firstly, I have worked on camera calibration for undistorting the images**
**Then, I have developed a computer vision pipeline that processes a group of test images then applied this pipeline to a test video stream.**

**In the following  writeup, I will be explaining each part of the project**

## Pipeline Architecture:
---
1. Calibrate camera using calibration images. This step is done only once, i.e., not repeated with the processing of each image
2. Undistort the image under processing
3. Apply gradient and color thresholdss (Sobelx, S channel, and L channel thresholds)
4. Cut-out any edges that are out of the lane lines region (**region of interest**)
5. Warp the image perspective to be in top-down (bird's eye) view
6. Search on lane lines in the image
   * Using the sliding window technique
   * If the previous lane lines detections are in reasonable area, search on lane lines based on previous frame
7. Calculae the lane curvature and vehicle distance from the lane center.  In this step we are also doing:
   * Drawing the lane line on blank image
   * Warping the blank back to original image space
   * Weighting the original image with the lane lines image

## Environment:
---
* Ubuntu 16.04 LTS
* Python 3.6.4
* OpenCV 3.1.0
* Anaconda 4.4.10

### 1. Calibrate camera using calibration images
---
A distortion is actually changing what the shape and size of these objects appear to be. So, we need to undistort images to reflect our real world surrondings.
To do so, a primary step is to measure that distortion, i.e., calibrate for distortion.
We can take pictures of known shapres then we will be able to detect and correct any distortion errors
We will use a chessboard as it s great for calibration because of its regular high contrast pattern, and aslo we know what undistorted flat chessboard looks like.
So, if we use our camera to take multiple pictures of a chessboard against a flat surface then weill be able to detect any distortion by looking at the difference between the apparent size and shape of the squares in these images, and the shape and size of the actual image.
Then we will use this information to calibrate our camera.
At this step we will be able to create a transform that maps these distorted points to undistorted points and finally we can undistort any image.

**In the following function we are doing the following for camera calibration:**
  * We map the coordinates of the corners in the 2-D image, which we will call `imgpoints` to the 3-D coordinates of the real chessboard corners which we will call `objpoints`. We will setup them as two empty arrays to hold these points
  * The object points will be the same as the known object coordinates of the chessboard corners for the 9 x 6 board. These points will be the 3-D coordinates x, y, and z from the top-left corner (0, 0, 0) to the botton right (8, 5, 0)
  * The z-coordinate will be zero for every object point. So, we will prepare these object points firstly be creating 9 x 6 point in an array each with 3 column for the x, y, and z coordinate. We will initialize all of these as zeros using numpy's `zeros` function
  * The z-coordinate will stay zero, so we will keept it as it is but for the first two columnsx and y we will use numpy's `mgrid` functionto generate the coordinates that we want.
  * `mgrid` returns the coordinates values for a given grid size and we will shape these coordinates back into two columns, one for x and other for y.
  * To create the image points, OpenCV gices an easy way to detect the chessboard corners with a function called `findChessboardCorners` that returns the corneers found in grayscale image. So, we will convert the image into grayscale and pass it to the `findChessboardCorners` function
  *  These `imgpoints` and `objpoints` are used to calibrate the camera using a function called `calibrateCamera`

Here is the function for getting the object and image points:
```python
def get_objectPoints_imgPoints_corners(calibration_images, nx, ny):
    """
    Descrition:
        This function helps calibrate the camera by getting the objects points and image points 
        for chessboard calibration images
    Paramteres:
        calibration_images: List of chessboard calibration images
        nx: number of inside corners in x
        ny: number of inside corners in y
    Output:
        objpoints: Object points
        imgpoints: Image points
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(calibration_images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            write_name = 'output_images/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            print("corner is not detected")
    cv2.destroyAllWindows()
    return objpoints, imgpoints
```

Then we use the object and image point to calibrate the camera. 
The following function takes the `objpoints` and `imgpoints` and returns the distortion coefficients `dist` and the camera matrix that we need to tranform 3-D object points to 2-D image points `mtx`

```python
def calibrate_camera(objpoints, imgpoints, img_size):
    """
    Descrition:
        This function calibrates the camera
    Paramteres:
        objpoints: Object points
        imgpoints: Image points
        img_size : size of the image
    Output:
        mtx: the camera matrix that we need to tranform 3-D object points to 2-D image points
        dist: the distortion coefficients
    """
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return mtx, dist
```
Here is a sample output of getting the chessboard corners for camera calibration
![corners](https://i.imgur.com/kfrFwZs.jpg)

### 2. Undistort the image under processing
---
We undistort the image using the OpenCV function `cv2.undistort`
The following function is used to undistorting images:
```python
def undistort(img, mtx, dist, test_images_mode=False):
    """
    Descrition:
        This function undistort the image under processing
    Paramteres:
        img: Distoted image to be undistorted
        mtx: the camera matrix that we need to tranform 3-D object points to 2-D image points
        dist: the distortion coefficients
    Output:
        undist: undistorted image
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
```

Here is an example of undistorted image
![chessboard_undist](https://i.imgur.com/bMdNSDP.png)

### 3. Apply gradient and color thresholdss (Sobelx, S channel, and L channel thresholds)
---
Here we need to detect the edges that are tending in some sorts to be vertical. This is achieved taking the image derivative in x-direction, i.e., the gradient in x-direction. This is called sobelx threshold.
But, the fact is RGB color spaces are not sufficient enough specially in different lighting conditions (like shadows) and some color (like yellow) can easily disappear when applying RGB thesholding.
So, depending on other color spaces may be useful for getting important information of an image to find the lane lines. 
One of othe color spaces is HLS (Hue, Lightness, and Saturation).
We can find using this color space that the S component of a channel detects the lane lines pretty well. This components stays fairly consistent under shadow or excessive brightness
Also, the L component isolates the lightness of each component which varies the most under different lighting conditions
So, if we use these channels, we should be able to detect different colors of the lane lines more reliabely that in R, G, and B color space only.
In the following function, we have used the sobelx, S channel, and L channel threshold for detecting the lane lines in different cirmustances.

```python
def sobelx_s_l_channels_thresholds(img, s_thresh=(170, 255), l_threshold=(150,200) , sx_thresh=(20, 100), test_images_mode=False):
    """
    Descrition:
        Applies the sobelx, s channel, and l channel threshold to an image
    Paramteres:
        img: Undistorted image to apply the thresholds to
        s_thresh: S channel threshold range
        l_threshold: L channel threshold range
        sx_thresh: Sobel x threshold range
    Output:
        combined_binary: Binary image that has edges detected
    """
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Threshold color channel (s channel)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
     # Threshold color channel (l channel)
    l_binary = np.zeros_like(l_channel)
    l_binary[(s_channel >= l_threshold[0]) & (s_channel <= l_threshold[1])] = 1
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( sxbinary, s_binary, l_binary)) 
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) |  (sxbinary == 1) | (l_binary == 1)] = 1
    combined_binary = np.asarray(combined_binary)
    if test_images_mode == True:
        if not hasattr(sobelx_s_l_channels_thresholds, "counter"):
            sobelx_s_l_channels_thresholds.counter = 0  # it doesn't exist yet, so initialize it
        sobelx_s_l_channels_thresholds.counter += 1
        
        write_name = 'output_images/01_color_thresholds_'+str(sobelx_s_l_channels_thresholds.counter)+'.jpg'
        (thresh, im_bw) = cv2.threshold(combined_binary, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(write_name, im_bw)
            
    return combined_binary
```

Here is a sample output of applying this function to one of the test images:
![color_threshold](https://i.imgur.com/ZjlGqQf.jpg)

### 4. Cut-out any edges that are out of the lane lines region **(region of interest)**
---
This function is used to have a better output for the histogram when applying the sliding window search of the next function. The regoin of interest function cuts out any other edges that may be detected out the lane lines region defined by the polygon vertices:
`bottom_left  = (140,image_shape[0])`
`up_left      = (550, 400)`
`up_right    = (780, 400)`
`bottom_right    = (1250,image_shape[0])`
where `image_shape[0]` is the y-coordinate of the image under processing.

Here is the function code for region of interest:
```python
def region_of_interest(image, image_shape, test_images_mode=False):
    """
    Descrition:
        Applies an image mask. Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
    Paramteres:
        image: The image to cut-out any other edges out if region of interest
        image_shape: shape of image under processing
    Output:
        An image that has the edges in the defined polygon and black everywhere else
    """
    # The following points are the vertices points of the polygon
    bottom_left  = (140,image_shape[0])
    up_left      = (550, 400)
    up_right    = (780, 400)
    bottom_right    = (1250,image_shape[0])
    vertices     = np.array([[bottom_left, up_left, up_right , bottom_right]], dtype=np.int32)
    #defining a blank mask to start with
    mask = np.zeros_like(image)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image_shape) > 2:
        channel_count = image_shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
```
Here is a sample output of applying this function to one of the test images:
![roi](https://i.imgur.com/EBkXdKJ.jpg)

### 5. Warp the image perspective to be in top-down (bird's eye) view
---
In an image, perspective is the phenomenon where objects appears smaller the farther away it is from a viewpoint like a camera, and parallel lines appear to converge to a point.
Here, we change our images to bird's eye view scene. This is really useful because some tasks like finding the curvature of a lane are easier to perform on a bird's eye view of an image. Also, it is useful to locate car's locatio directly with a map, since maps display roads from a top-down view.

To create a perspective transfrom:
  * We firstly select four points that define a the lane lines in our image which is trapezoidal shape. `src`
  * Then we select where we want there four points to appear in the warped image. These points basically represent a rectangle forming a line in a bird's eye view. `dst`
  * Then we use OpenCV function `cv2.getPerspectiveTransform` to calculate the transform matrix that maps the original image to the warped image.
  * We can get the inverse perspective transform to unwarp by just switching the `src` and `dst`
  * To apply the perspective transform matrix to the original image, we use the OpenCV function `cv2.warpPerspective` which returns the warped image.

> Note: when the input to this function is the chessboard image, we use automatice detection to `src` and `dst` based on the chessboard corners.

This is the function python code for warping the image perspective:
```python
def warp_image_perspective(undist, nx=9, ny=6, test_images_mode=False):
    """
    Descrition:
        Applies perspective transfrom to get the image in bird's eye view
    Paramteres:
        undist: The undistorted image
        nx: number of inside corners in x
        ny: number of inside corners in y
    Output:
        warped_image: Image in bird's eye (top-down) view 
        M: Tranformation matrix from original image to bird's eye view 
        Minv: Inverse matrix from bird's eye view to original view
    """
    image_shape = undist.shape
    img_size = (undist.shape[1] , undist.shape[0])
    ret, corners = cv2.findChessboardCorners(undist, (nx, ny), None)
    offset = 50
    if ret == True:
        top_left = corners[0]
        top_right = corners[nx-1]
        bottom_right = corners[-1]
        bottom_left = corners[-nx]
        src = np.float32([top_left,
                          top_right,
                          bottom_right,
                          bottom_left])
        dst = np.float32([[offset, offset],
                  [img_size[0]-offset, offset],
                  [img_size[0]-offset, img_size[1]-offset],
                  [offset, img_size[1]-offset]])
    else:
        # The following points are the vertices points of the polygon
        src_bottom_left  = (200,image_shape[0]-offset)
        src_top_left      = (580, 450)
        src_top_right     = (700, 450)
        src_bottom_right = (1100,image_shape[0]-offset)
        src = np.float32([src_top_left,
                          src_top_right,
                          src_bottom_right,
                          src_bottom_left])
        
        #The following points are the vertices points of the rectangle
        dst_bottom_left  = (300,image_shape[0])
        dst_top_left     = (300, 0)
        dst_top_right    = (950, 0)
        dst_bottom_right = (950,image_shape[0])
        dst = np.float32([dst_top_left,
                          dst_top_right,
                          dst_bottom_right,
                          dst_bottom_left])        

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_image = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    warped_image = np.asarray(warped_image)
    return warped_image, M, Minv
```
Here is an outout of chessboard image warping:
![chessboard_warping](https://i.imgur.com/FK9wKDB.png)

Here is a sample output of applying this function to one of the test images:
![test_image_warping](https://i.imgur.com/a92exLp.jpg)

### 6. Search on lane lines in the image
---
   * Using the sliding window technique
   * If the previous lane lines detections are in reasonable area, search on lane lines based on previous frame

After applying calibration, undistorting, thersholding, regoin of interest, and perspective transform to a road image, we now have an image where lane lines standout clearly. However, we still need to decide explicitly which pixels are part of the lines and which belong to the left and which belong to the right line.

#### Using the sliding window technique
  * First, we take the histogram to the lower half of the image
  * We can use the values from this histogram as a starting point for where to search for lines
  * From these point, we can use sliding window placed around the line centers to find and follow the lines up to the top of the frame to get left and right x and y pixels of the lane lines
  * After getting these points, we can fit a second-degree polynomial using `np.polyfit(lefty, leftx, 2)` and `np.polyfit(righty, rightx, 2)`
  * Then we plot these lane lines detected

In the following is the python code for this function:
```python
def search_for_lane_lines(warped_roi_image, test_images_mode=False):
    """
    Descrition:
        Searches for lane lines in warped image based on sliding window technique
    Paramteres:
        warped_roi_image: The warped image
    Output:
        out_img: Output image that has the lane lines detected
        leftx: Left lane line pixels x-position
        lefty: Left lane line pixels y-position
        rightx: right lane line pixels x-position
        righty: right lane line pixels y-position 
        left_fit: Coeffecients of the second-degree ploynomial fit for the left lane
        right_fit: Coeffecients of the second-degree ploynomial fit for the right lane
        left_fitx: Left lane x values used for plotting the lane line
        right_fitx: Right lane x values used for plotting the lane line
        ploty: Y values used for plotting the lane line
    """
    histogram = np.sum(warped_roi_image[warped_roi_image.shape[0]//2:,:], axis=0)
    out_img = np.dstack((warped_roi_image, warped_roi_image, warped_roi_image))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_roi_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100
    
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds  = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_roi_image.shape[0] - (window+1)*window_height
        win_y_high = warped_roi_image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_roi_image.shape[0]-1, warped_roi_image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return out_img, leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty
```

Here is the output of applying this function to one of the test images:
![sliding_window](https://i.imgur.com/UUoEoA3.jpg)

#### If the previous lane lines detections are in reasonable area, search on lane lines based on previous frame
  * Based on the left and right lane lines detections have drifts to specific values or not, we can depend on plane lines detected from previous frame applying a margin value.
  * After applying this margin to previous lane lines, we can extract the right and left x and y pixels for lane lines.
  * Like the previous function, we can fit a second-degree polynomial using `np.polyfit(lefty, leftx, 2)` and `np.polyfit(righty, rightx, 2)`
  * Then we plot these lane lines detected
  
> I will be commenting and explaining more about the specific values drifts (filtering to re-do the sliding window or depend on previous frame) at the `algorithm_pipline` function

The following is the python code for this function:
```python
def search_for_lane_lines_based_on_previous_frame(warped_roi_image, left_fit, right_fit):
    """
    Descrition:
        Searches for lane lines in warped image based on previous frame
    Paramteres:
        warped_roi_image: The warped image
        left_fit: Coeffecients of the second-degree ploynomial fit for the left lane from previous frame
        right_fit: Coeffecients of the second-degree ploynomial fit for the right lane from previous frame
    Output:
        result: Output image that has the lane lines detected
        leftx: Left lane line pixels x-position
        lefty: Left lane line pixels y-position
        rightx: right lane line pixels x-position
        righty: right lane line pixels y-position 
        left_fit: Coeffecients of the second-degree ploynomial fit for the left lane
        right_fit: Coeffecients of the second-degree ploynomial fit for the right lane
        left_fitx: Left lane x values used for plotting the lane line
        right_fitx: Right lane x values used for plotting the lane line
        ploty: Y values used for plotting the lane line
    """
    # Assume you now have a new warped binary image 
    # from the next frame of video 
    # It's now much easier to find line pixels!
    nonzero = warped_roi_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                                                         left_fit[1]*nonzeroy + left_fit[2] + margin)))
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                                                           right_fit[1]*nonzeroy + right_fit[2] + margin)))
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_roi_image.shape[0]-1, warped_roi_image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped_roi_image, warped_roi_image, warped_roi_image))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return result, leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty
```

### 7. Calculae the lane curvature and vehicle distance from the lane center.
---
This function also containts:
  * Draw the lane line on blank image
  * Warp the blank back to original image space using inverse perspective matrix (inv_perspective_M)
  * Weighting the original image with the lane lines image

This function is used to:
  * determine the lane curvature in real world space (not in pixels space)
  * determine the distace of vehicle position from lane center
  * weight the lane lines on the undistorted image


#### Determine the lane curvature in real world space (not in pixels space)
---
  * To convert from pixels space to meters, the following equations are used for both x and y, 
  `ym_per_pix = 30/720` and `xm_per_pix = 3.7/700`
  * As we have done when searching for lane lines, we fit a second-degree polynomial but for real world values not for pixel values
  * The radius of curvature is define as:
![rad_cur](https://i.imgur.com/fCv3cbS.png)
where we can get `A` and `B` from fitting the second-degree ploynomial.

  * Implementation of the previous equation for the left curvature will look like:  
  `left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])`
  
  
  * Implementation of the previous equation for the right curvature will look like:  
  `right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])`
  
  
  * Estimating the radius of curvature as the mean of the right and left curvatures.

#### Determine the distace of vehicle position from lane center
---
  * For determining the distace from the lane center we need to determine the vehicle position and the lane center
  * For the vehicle position, I will assume it is in the center of the x of the image, assuming that the front camera is mounted in the center of the hood.
  * For the estimated lane center, I get an average value for the max and min left and right x value. Then I get their mean to estimate the lane center.
  * Subtracting the vehicle postion from the estimated lane center, we can get the distance in pixels value
  * Converting this distance from pixels to meters, we get the vehicle's distance from the lance center
  
#### Weight the lane lines on the undistorted image
---
  * At the end, draw lane lines on blank image
  * Warp the blank back to original image space using inverse perspective matrix (inv_perspective_M)
  * Then we weight this image to the undistorted original image.
  * We put the text of radius of curvature and vehicle distance from lane center

Here is the python code code:
```python
def calculate_lane_curvature_and_vehicle_position(undistorted_image, lane_lines_image, leftx, lefty, ploty, rightx, righty,left_fitx, right_fitx, inv_perspective_M, test_images_mode=False):
    """
    Descrition:
        Calculates the lane curvature and vehicle distance from lane center.
    Paramteres:
        undistorted_image: undistorted original image
        lane_lines_image: Image that has lane lines 
        leftx: Left lane line pixels x-position
        lefty: Left lane line pixels y-position
        ploty: Y values used for plotting the lane line
        rightx: right lane line pixels x-position
        righty: right lane line pixels y-position 
        left_fitx: Left lane x values used for plotting the lane line
        right_fitx: Right lane x values used for plotting the lane line
        inv_perspective_M: Inverse transformation matrix from bird's view to original view
    Output:
        result: Output image with lane lines determined, radius of curvature, and vehicle distance from lane center
    """
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image    
    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # Create an image to draw the lines on
    color_warp = np.zeros_like(lane_lines_image).astype(np.uint8)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (inv_perspective_M)
    newwarp = cv2.warpPerspective(color_warp, inv_perspective_M, (lane_lines_image.shape[1], lane_lines_image.shape[0]))
    
    result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)
    
    lane_curvature = (left_curverad + right_curverad) / 2
    curve_str = "Lane Curvature " + '{:04.3f}'.format(lane_curvature) + " m" # + " m .. Right Line Curvature " + str(right_curverad)+ " m"
    if lane_curvature > 2000:
        curve_str = curve_str + " (it seems a straight lane !) "
    veh_center_pos = lane_lines_image.shape[1] / 2
    left_pnt = ( np.max(leftx) + np.min(leftx) ) / 2
    right_pnt = ( np.max(rightx) + np.min(rightx) ) / 2
    lane_center =  ( left_pnt + right_pnt ) / 2
    distance_from_lane_center = (lane_center - veh_center_pos) * xm_per_pix
    pos_str = "Distace from lane center " + '{:04.3f}'.format(distance_from_lane_center) + " m"
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,100)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(result,curve_str,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    
    bottomLeftCornerOfText = (10,200)
    cv2.putText(result,pos_str,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    return result
```

Here is a sample output of applying this function to one of the test images:
![cur_pos](https://i.imgur.com/jXkf7FD.jpg)


### Algorithm Pipeline:
It is function that wraps and calls all the functions needed to detect the lane lines on the road of an image
Here some important notes about this function
   * This function uses the lane feature class to keep some improtant properties of the lane line
   * Some properties are very improtant to be kept specially when finding the lane lines based on previous frame. So, we need so feature of the previously detected lane to feed to the function
   * I applied a filter to decide whether this previously detected lane I can depend on finding new lane line or I need to re-do sliding window. This filteration criteria is as follows `if( (np.min(leftx) < 200) | (np.max(rightx) > 1000) )`
   * It seems it is in some sorts a very hard filteration criteria. Yes, but I see it is more beneficial that getting the lane line jumping out of the boundries.

Here is the python code for the algorithm pipeline:
```python
images_count = 0
lane = Lane_Features()
def algorithm_pipeline(image, test_images_mode):
    """
    Descrition:
        Wrapper for all the functions needed to detect lane line on the road
    Paramteres:
        image: Image under processing
        test_images_mode: A flag to differentiate between test images and video processing
    Output:
        result_image: An image with lane line detected, curvature, and vehicle distance from lane center        
    """
    global images_count
    activate_sliding_window = False
    leftx = lane.leftx
    lefty = lane.lefty
    rightx = lane.rightx
    righty = lane.righty
    left_fit = lane.left_fit
    right_fit = lane.right_fit
    left_fitx = lane.left_fitx
    right_fitx = lane.right_fitx
    ploty = lane.ploty
    undistorted_image = undistort(image, mtx, dist, test_images_mode=test_images_mode)
    combined_binary_image = sobelx_s_l_channels_thresholds(undistorted_image, test_images_mode=test_images_mode)
    roi_image = region_of_interest(combined_binary_image, image.shape, test_images_mode=test_images_mode)
    warped_perspective_roi_image, perspective_M, inv_perspective_M = warp_image_perspective(roi_image, test_images_mode=test_images_mode)
    try:
        if( (np.min(leftx) < 200) | (np.max(rightx) > 1000) ):
            activate_sliding_window = True
    except ValueError:
        activate_sliding_window = True
        pass
    if( (test_images_mode == True) | (activate_sliding_window == True) ):
        output_lane_lines_image, leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty = search_for_lane_lines(warped_perspective_roi_image, test_images_mode=test_images_mode)
        print("Sliding Window Done")
        
    else:
        output_lane_lines_image, leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty = search_for_lane_lines_based_on_previous_frame(warped_perspective_roi_image, left_fit, right_fit)
        print("Search based on prev frame Done")
    result_image = calculate_lane_curvature_and_vehicle_position(undistorted_image, output_lane_lines_image, leftx, lefty, ploty, rightx, righty,left_fitx, right_fitx, inv_perspective_M, test_images_mode=test_images_mode) 
    images_count = images_count + 1
    lane.leftx = leftx
    lane.lefty = lefty
    lane.rightx = rightx
    lane.righty = righty
    lane.left_fit = left_fit
    lane.right_fit = right_fit
    lane.left_fitx = left_fitx
    lane.right_fitx = right_fitx
    lane.ploty = ploty
    activate_sliding_window = False
    return result_image
```


## Conclusion
---
  * The algorithm pipeline was acceptedly finding lane lines even if differnt conditions
  * The pipeline gets back on track when tough movement occurs
  * However, we can notice that filteration criteria for activiating the sliding window is tough enough that actuivates the sliding window in much cases. So improving this criteria is one of the improvements
  * Invistigation of using other color thresholding, like using the H compenent for improving the lane finding in more challenging conditions. More even, more investigating the used thresholds for more complex situations
  * Improving the distance from lane center output to determine where it is distannt from left or right of the center