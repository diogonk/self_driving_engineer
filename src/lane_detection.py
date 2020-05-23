#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def empty(a):
    pass

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    #plt.imshow(mask)
    #plt.show()

    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, lanes_average=[]):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    averaged_lines = slope_intercept(line_img, lines)
    if(check_valid_lane(averaged_lines)):
        if not lanes_average:
            for x in range(0, 20):
                print("new average lines")
                lanes_average.append(averaged_lines)
        else:
            print("appending average line")
            lanes_average[0:18] = lanes_average[1:19]
            lanes_average[19] = averaged_lines
    draw_lines(line_img, np.average(lanes_average,axis=0).astype(int))
    return line_img

def check_valid_lane(lines):
    for line in lines:
        for x1,y1,x2,y2 in line:
            if(x1 == 0 & y1 == 0 & x2 == 0 & y2 == 0):
                return False
    return True

# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    left_line = [[0, 0, 0, 0]]
    right_line = [[0, 0, 0, 0]]
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            if(m != 0): #not horizontal line
                if m < 0: 
                    left_fit.append((m, b))# is a left line "/"
                else:
                    right_fit.append((m, b))# is a right line "\"
    if left_fit:
        left_fit_average  = np.average(left_fit, axis=0)
        left_line  = create_line(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = create_line(image, right_fit_average)

    averaged_lines = [left_line, right_line]
    return averaged_lines

def create_line(img, line):
    m, b = line
    line_points = [np.ndarray(shape=(1,2), dtype=np.int32)]
    y1 = (image.shape[0])
    y2 = int(0.57*imshape[0])
    x1 = int((y1 - b)/m)
    x2 = int((y2 - b)/m)
    line_points = [x1, y1, x2, y2]
    return [line_points]


#reading in an image
images_path = []
images_path.append('test_images/solidWhiteCurve.jpg')
images_path.append('test_images/solidWhiteRight.jpg')
images_path.append('test_images/solidYellowCurve.jpg')
images_path.append('test_images/solidYellowCurve2.jpg')
images_path.append('test_images/solidYellowLeft.jpg')
images_path.append('test_images/whiteCarLaneSwitch.jpg')
images_path.append('test_images/savedImage.jpg')
#image = cv2.imread(path[0])
#cv2.imshow("Image", image)
videos_path = []
videos_path.append('test_videos/solidWhiteRight.mp4')
videos_path.append('test_videos/solidYellowLeft.mp4')
videos_path.append('test_videos/challenge.mp4')

lanes_average = []

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
#line_image = np.copy(image)*0 # creating a blank to draw lines on
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1*np.pi/180 # angular resolution in radians of the Hough grid
threshold = 25     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 35 #minimum number of pixels making up a line
max_line_gap = 30    # maximum gap in pixels between connectable line segments

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5

# Define our parameters for Canny and apply
low_threshold = 170
high_threshold = 200


cap = cv2.VideoCapture(videos_path[1])

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()
    #plt.imshow(lines_edges)
    if cv2.waitKey(1) & 0xFF == ord('q') or ret != True:
        break

# Read in and grayscale the image
    gray = grayscale(image)

    blur_gray = gaussian_blur(gray, kernel_size)

    edges = canny(blur_gray, low_threshold, high_threshold)
    cv2.imshow("edges", edges)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    #vertices = np.array([[(0,imshape[0]),(450, 300), (490, 300), (imshape[1],imshape[0])]], dtype=np.int32)
    vertices = np.array([[(0.15*imshape[1],imshape[0]),(0.47*imshape[1], 0.57*imshape[0]), (0.51*imshape[1], 0.57*imshape[0]), (0.9*imshape[1],imshape[0])]], dtype=np.int32)
    masked_image = region_of_interest(edges, vertices)
    cv2.imshow("masked_image", masked_image)



    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_image, rho, theta, threshold,
            min_line_length, max_line_gap,lanes_average)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    lines_edges = weighted_img(image, line_image) 
    cv2.imshow("image", lines_edges)


#    plt.show()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
