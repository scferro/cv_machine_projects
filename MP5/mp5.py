import numpy as np
import cv2


### INPUTS ###
file = 'lena.bmp'


### FUNCTIONS ###
def GaussSmoothing(img, kernel_size, sigma):
    # Create kernel
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)/2)**2 + (y-(kernel_size-1)/2)**2) / (2*sigma**2)), (kernel_size, kernel_size))
    
    output_img = apply_kernel(img, kernel)

    return output_img

def ImageGradient(img):
    # Calculate the gradient using the Sobel method
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gx = apply_kernel(img, sobel_x)
    gy = apply_kernel(img, sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    return magnitude, direction

def FindThreshold(Mag, percentageOfNonEdge):
    histogram, bin_edges = np.histogram(Mag, bins=np.max(Mag), density=False)

    # Make histogram
    num_pixels_above_thresh = Mag.size * percentageOfNonEdge

    # Determine the threshold from the histogram
    count = 0
    for i in range(len(histogram) - 1, -1, -1):
        count += histogram[i]
        if count >= num_pixels_above_thresh:
            thresh_index = i
            break

    # Find the corresponding gradient value
    t_high = bin_edges[thresh_index]

    t_low = 0.5 * t_high
    return t_low, t_high

def NonmaximaSupress(Mag, Theta):
    M, N = Mag.shape
    output_mag = np.zeros((M, N), dtype=np.float32)
    angles = Theta * 180. / np.pi
    angles[angles < 0] += 180

    # Quantization
    quantized_angles = np.round(angles / 45) * 45
    quantized_angles = quantized_angles % 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q_angle = quantized_angles[i, j] % 180
            if q_angle == 0: 
                g1, g2 = Mag[i, j-1], Mag[i, j+1]
            elif q_angle == 45: 
                g1, g2 = Mag[i-1, j+1], Mag[i+1, j-1]
            elif q_angle == 90:  
                g1, g2 = Mag[i-1, j], Mag[i+1, j]
            elif q_angle == 135: 
                g1, g2 = Mag[i-1, j-1], Mag[i+1, j+1]

            # Linear interpolation
            gx = (g1 + g2) / 2

            if Mag[i, j] > gx:
                output_mag[i, j] = Mag[i, j]
            else:
                output_mag[i, j] = 0
            
    return output_mag

def EdgeLinking():
    pass

def apply_kernel(img, kernel):
    # Get kernel and image sizes
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = img.shape

    # Create an output array
    output_img = np.zeros_like(img)

    # Pad the input image to handle borders
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Convolution operation
    for i in range(image_height):
        for j in range(image_width):
            output_img[i, j] = np.sum(kernel * padded_image[i:i + kernel_height, j:j + kernel_width])

    return output_img

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

gauss_image = GaussSmoothing(image, 5, 1.0)
show_image(gauss_image)