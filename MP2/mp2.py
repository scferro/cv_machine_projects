import numpy as np
import cv2

### INPUTS ###
kernel1 = np.array([[1,1,1],
                    [1,1,1],
                    [1,1,1]])

kernel2 = np.array([[0,1,1,1,0],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [0,1,1,1,0]])

kernel3 = np.array([[0,0,1,1,1,0,0],
                    [0,1,1,1,1,1,0],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],
                    [0,1,1,1,1,1,0],
                    [0,0,1,1,1,0,0]])


### FUNCTIONS ###
def Erosion(img_ero, kernel):
    # Create empty image
    img_out = np.zeros(img_ero.shape)

    # Iterate through pixels in input image
    for y in range(img_ero.shape[0]):
        for x in range(img_ero.shape[1]):
            all_valid = True
            # Iterate through kernel pixels
            for y_ker in range(kernel.shape[0]):
                for x_ker in range(kernel.shape[1]):
                    # Check if all overlapping pixels are >0
                    y_ = int(-kernel.shape[0] / 2 + y_ker + y) + 1
                    x_ = int(-kernel.shape[1] / 2 + x_ker + x) + 1
                    if y_ >= 0 and x_ >= 0 and y_ < img_ero.shape[0] and x_ < img_ero.shape[1] and kernel[y_ker, x_ker] > 0 and all_valid:
                        if img_ero[y_, x_] > 0:
                            pass
                        else:
                            # If pixel not 0, set all_valid to False
                            all_valid = False
            # If all pixels masked by kernel are >0, set img_out pixel to 255
            if all_valid:
                img_out[y, x] = 255
                
    return img_out


def Dilation(img_dil, kernel):
    # Create empty image
    img_out = np.zeros(img_dil.shape)

    # Iterate through pixels in input image
    for y in range(img_dil.shape[0]):
        for x in range(img_dil.shape[1]):
            # Check if pixel is 1
            if img_dil[y, x] > 0:
                # Iterate through kernel pixels
                for y_ker in range(kernel.shape[0]):
                    for x_ker in range(kernel.shape[1]):
                        # Update pixels in image out based on kernel
                        y_ = int(-kernel.shape[0] / 2 + y_ker + y) + 1
                        x_ = int(-kernel.shape[1] / 2 + x_ker + x) + 1
                        if y_ >= 0 and x_ >= 0 and y_ < img_dil.shape[0] and x_ < img_dil.shape[1] and kernel[y_ker, x_ker] > 0:
                            img_out[y_, x_] = 255

    return img_out


def Opening(img_open, kernel):
    # Erode image
    img_ero = Erosion(img_open, kernel)
    # Dilate image
    img_out = Dilation(img_ero, kernel)

    return img_out


def Closing(img_close, kernel):
    # Dilate image
    img_dil = Dilation(img_close, kernel)
    # Erode image
    img_out = Erosion(img_dil, kernel)

    return img_out


def Boundary(img_bound, kernel):
    # Create empty image
    img_out = np.zeros(img_bound.shape)

    # Erode image
    img_ero = Erosion(img_bound, kernel)

    # Iterate thru pixels
    for y in range(img_bound.shape[0]):
        for x in range(img_bound.shape[1]):
            # Set pixel to 255 in output if it is white in input image but black in eroded image
            if img_bound[y, x] > 0 and img_ero[y, x] == 0:
                img_out[y, x] = 255
            else: 
                pass

    return img_out



def show_image(img_show, filename, str=''):
    cv2.imshow('Grayscale Image', img_show)
    # Wait for any key to be pressed
    print('Displaying image. Press any key to continue...')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if str != '':
        print('Saving image...')
        file_out = filename[0:-4] + '_' + str + '_out.png'
        cv2.imwrite(file_out, img_show)



### MAIN CODE ###
# gun.bmp
# Import file
file = 'gun.bmp'
image = cv2.imread(file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show_image(image, file)

# Erosion
image_erosion = Erosion(image, kernel1)
show_image(image_erosion, file, 'ero')

# Dilation
image_dilation = Dilation(image, kernel1)
show_image(image_dilation, file, 'dil')

# Opening
image_open = Opening(image, kernel1)
show_image(image_open, file, 'open')

# Closing
image_close = Closing(image, kernel1)
show_image(image_close, file, 'close')

# Remove Noise
image_clean = Closing(image, kernel3)
image_clean = Opening(image_clean, kernel2)
show_image(image_clean, file, 'clean')

# Boundary
image_bound = Boundary(image_clean, kernel1)
show_image(image_bound, file, 'bound')

# palm.bmp
# Import file
file = 'palm.bmp'
image = cv2.imread(file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show_image(image, file)

# Erosion
image_erosion = Erosion(image, kernel1)
show_image(image_erosion, file, 'ero')

# Dilation
image_dilation = Dilation(image, kernel1)
show_image(image_dilation, file, 'dil')

# Opening
image_open = Opening(image, kernel1)
show_image(image_open, file, 'open')

# Closing
image_close = Closing(image, kernel1)
show_image(image_close, file, 'close')

# Remove Noise
image_clean = Closing(image, kernel3)
image_clean = Opening(image_clean, kernel2)
show_image(image_clean, file, 'clean')

# Boundary
image_bound = Boundary(image_clean, kernel1)
show_image(image_bound, file, 'bound')