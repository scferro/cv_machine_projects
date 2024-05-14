import numpy as np
import cv2


### INPUTS ###
files = ['gun1.bmp', 'joy1.bmp', 'lena.bmp', 'pointer1.bmp', 'test1.bmp']


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
    max = int(np.max(Mag)) + 1
    # Make histogram
    histogram, bin_edges = np.histogram(Mag, bins=max, density=False)

    num_pixels_below_thresh = Mag.size * percentageOfNonEdge

    # Determine the threshold from the histogram
    count = 0
    for i in range(len(histogram)):
        count += histogram[i]
        if count >= num_pixels_below_thresh:
            thresh_index = i
            break

    # Find the corresponding gradient value
    T_high = bin_edges[thresh_index]

    T_low = 0.5 * T_high
    return T_low, T_high

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

def EdgeLinking(Mag, T_low, T_high):
    strong_edges = np.zeros(Mag.shape, dtype=np.uint8)
    strong_edges[Mag > T_high] = 1

    weak_edges = np.zeros(Mag.shape, dtype=np.uint8)
    weak_edges[Mag > T_low] = 1

    # Define the structure for the 8-connected neighborhood
    neighborhood = np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]])

    while True:
        # Create a copy of strong edges to check for changes after iteration
        old_strong_edges = np.copy(strong_edges)

        # Check if strong edges are adjacent to weak edges
        for i in range(1, strong_edges.shape[0] - 1):
            for j in range(1, strong_edges.shape[1] - 1):
                if weak_edges[i, j] and np.any(neighborhood * strong_edges[i-1:i+2, j-1:j+2]):
                    strong_edges[i, j] = 1

        print(np.count_nonzero(old_strong_edges))
        print(np.count_nonzero(strong_edges))

        # Break the loop if no new strong edges were added
        if np.array_equal(old_strong_edges, strong_edges):
            break

    img_out = (strong_edges * 255).astype(np.uint8)
    
    return img_out

def apply_kernel(img, kernel, operation='multiply'):
    # Get kernel and image sizes
    kernel_size, _ = kernel.shape
    image_height, image_width = img.shape

    # Create an output array
    output_img = np.zeros_like(img)

    # Pad the input image to handle borders
    pad_height = kernel_size // 2
    pad_width = kernel_size // 2
    padded_image = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Convolution operation
    if operation=='multiply':
        for i in range(image_height):
            for j in range(image_width):
                output_img[i, j] = np.sum(kernel * padded_image[i:i + kernel_size, j:j + kernel_size])
    elif operation=='check_all':
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output_img[i, j] = (kernel * padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]).sum()

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
for file in files:
    # Import file
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image(image)

    # Perform Gaussian smoothing
    gauss_image = GaussSmoothing(image, 7, 5.0)
    show_image(gauss_image, file, 'gauss_out')

    # Get gradient
    Mag, Theta = ImageGradient(gauss_image)
    print('Calculated gradient.')
    T_low, T_high = FindThreshold(Mag, 0.999)
    print('Found threshold values.')
    Mag = NonmaximaSupress(Mag, Theta)
    print('Non-maxima supressed.')

    # Display Mag
    min_val = np.min(Mag)
    max_val = np.max(Mag)
    Mag_disp = (Mag - min_val) / (max_val - min_val)
    print('Displaying normalized magnitude of gradient.')
    show_image(Mag_disp, file, 'mag_out')

    # Display high threshold
    Mag_thresh_hi = np.zeros(Mag.shape, dtype=np.uint8)
    Mag_thresh_hi[Mag > T_high] = 255
    print('Displaying high threshold image')
    show_image(Mag_thresh_hi, file, 'high_out')

    # Display low threshold
    Mag_thresh_lo = np.zeros(Mag.shape, dtype=np.uint8)
    Mag_thresh_lo[Mag > T_low] = 255
    print('Displaying low threshold image.')
    show_image(Mag_thresh_lo, file, 'low_out')

    # Get edges using magnitudes and threshold values
    edges = EdgeLinking(Mag, T_low, T_high)
    print('Displaying linked edges.')
    show_image(edges, file, 'edges_out')

    # Compare to OpenCV
    edges_cv2 = cv2.Canny(image, threshold1=T_low, threshold2=T_high)
    print('Displaying OpenCV edge detection output.')
    show_image(edges_cv2, file, 'edges_cv2_out')