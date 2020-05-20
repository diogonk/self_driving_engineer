import cv2  #bringing in OpenCV libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


image = mpimg.imread('resources/exit-ramp.jpg')
#plt.imshow(image)
#plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion
#plt.imshow(gray, cmap='gray')
#plt.show()


# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

plt.imshow(edges, cmap='Greys_r')
plt.show()
