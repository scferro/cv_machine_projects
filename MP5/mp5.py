import numpy as np
import cv2


### INPUTS ###
file = 'lena.bmp'


### FUNCTIONS ###
def GaussSmoothing(img, kernel_size, sigma):
    # Create kernel
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)/2)**2 + (y-(kernel_size-1)/2)**2) / (2*sigma**2)), (kernel_size, kernel_size))

def ImageGradient(S):
    pass 

def FindThreshold(Mag, percentageOfNonEdge):
    pass

def NonmaximaSupress(Mag, Theta, method):
    pass

def EdgeLinking():
    pass

def show_image(img_show, filename='    ', str=''):
    cv2.imshow('Image', img_show)
    # Wait for any key to be pressed
    print('Displaying image. Press any key to continue...')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if str != '':
        print('Saving image...')
        file_out = filename[0:-4] + '_' + str + '.png'
        cv2.imwrite(file_out, img_show)



### MAIN CODE ###
# Import file
image = cv2.imread(file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show_image(image)

GaussSmoothing(image, 5, 0.5)