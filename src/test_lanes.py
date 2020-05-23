#importing some useful packages
import numpy as np
import cv2
import math

def empty(a):
    pass

#reading in an image
images_path = []
images_path.append('test_images//solidWhiteCurve.jpg')
images_path.append('test_images//solidWhiteRight.jpg')
images_path.append('test_images//solidYellowCurve.jpg')
images_path.append('test_images//solidYellowCurve2.jpg')
images_path.append('test_images//solidYellowLeft.jpg')
images_path.append('test_images//whiteCarLaneSwitch.jpg')
images_path.append('test_images/savedImage.jpg')


# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
#rho = 1 # distance resolution in pixels of the Hough grid
#theta = 1*np.pi/180 # angular resolution in radians of the Hough grid
#threshold = 5     # minimum number of votes (intersections in Hough grid cell)
#min_line_length = 25 #minimum number of pixels making up a line
#max_line_gap = 5    # maximum gap in pixels between connectable line segments


rho = 2 # distance resolution in pixels of the Hough grid
theta = 1 # angular resolution in radians of the Hough grid
threshold = 25     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 35 #minimum number of pixels making up a line
max_line_gap = 30    # maximum gap in pixels between connectable line segments

# Define a kernel size and apply Gaussian smoothing
kernel_size = 3

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("image", "TrackBars", 0, 6, empty)
cv2.createTrackbar("rho", "TrackBars", rho, 60, empty)
cv2.createTrackbar("theta", "TrackBars", theta, 60, empty)
cv2.createTrackbar("threshold", "TrackBars", threshold, 100, empty)
cv2.createTrackbar("min_len", "TrackBars", min_line_length, 60, empty)
cv2.createTrackbar("max_gap", "TrackBars", max_line_gap, 100, empty)
cv2.createTrackbar('threshold1', 'TrackBars', low_threshold, 255, empty)
cv2.createTrackbar('threshold2', 'TrackBars', high_threshold, 255, empty)
cv2.createTrackbar('filter_size', 'TrackBars', kernel_size, 20, empty)
while True:

    image = cv2.imread(images_path[cv2.getTrackbarPos("image", "TrackBars")])

    # Read in and grayscale the image
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = cv2.getTrackbarPos("filter_size", "TrackBars")
    kernel_size += (kernel_size + 1) % 2
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # Define our parameters for Canny and apply
    low_threshold = cv2.getTrackbarPos("threshold1", "TrackBars")
    high_threshold = cv2.getTrackbarPos("threshold2", "TrackBars")
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0.15*imshape[1],imshape[0]),(0.47*imshape[1], 0.57*imshape[0]), (0.51*imshape[1], 0.57*imshape[0]), (0.9*imshape[1],imshape[0])]], dtype=np.int32)
    #vertices = np.array([[(0,imshape[0]),(450, 300), (490, 300), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    line_image = np.copy(image)*0 # creating a blank to draw lines on
    rho = cv2.getTrackbarPos("rho", "TrackBars")
    theta = cv2.getTrackbarPos("theta", "TrackBars")*np.pi/180
    threshold = cv2.getTrackbarPos("threshold", "TrackBars")
    min_line_length = cv2.getTrackbarPos("min_len", "TrackBars")
    max_line_gap = cv2.getTrackbarPos("max_gap", "TrackBars")

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0,0,255), 3)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
    cv2.imshow("image", lines_edges)
    cv2.waitKey(1)
cv2.destroyAllWindows()