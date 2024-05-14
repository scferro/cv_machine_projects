import numpy as np
import cv2
import matplotlib.pyplot as plt


### INPUTS ###
files = ['test.bmp', 'test2.bmp', 'input.bmp']
# files = ['test.bmp']


### FUNCTIONS ###
def HoughTransform(edges, threshold):
    # Define the Hough space
    theta = np.deg2rad(np.arange(-90, 90))
    # Find the maximum rho from the diagonal length of the image
    diag_len = int(np.sqrt(edges.shape[0]**2 + edges.shape[1]**2)) 
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Create accumulator to count votes
    accumulator = np.zeros((2 * diag_len, len(theta)), dtype=np.int32)

    # Iterate through edge pixels, calculate rho for each pixel
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] > 0:
                for t_idx, t in enumerate(theta):
                    rho = int(x * np.cos(t) + y * np.sin(t))
                    accumulator[rho + diag_len, t_idx] += 1

    # Select lines with more votes than the threshold
    result = np.where(accumulator > threshold)
    # Create output image
    image_out = np.zeros_like(edges)

    # Iterate through results, add lines to image
    for i in range(len(result[0])):
        rho_idx = result[0][i]
        theta_idx = result[1][i]
        rho = rhos[rho_idx]
        th = theta[theta_idx]
        a = np.cos(th)
        b = np.sin(th)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_out, (x1, y1), (x2, y2), 255, 2)

    return image_out, accumulator

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

def plot_accumulator(accumulator):
    # Plotting the accumulator
    plt.figure(figsize=(10, 10))
    plt.imshow(accumulator, aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title('Hough Transform Accumulator')
    plt.xlabel('Theta (angle)')
    plt.ylabel('Rho (distance from origin)')
    plt.show()


### MAIN CODE ###
for file in files:
    # Import file
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('Displaying original image.')
    show_image(image)

    # Extract edges with canny edge detection (refer to MP5 for specifics on Canny Edge Detection)
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    print('Displaying edges.')
    show_image(edges)

    # Get edges using magnitudes and threshold values
    hough, accumulator = HoughTransform(edges, 50)
    print(np.max(hough))
    # plot_accumulator(accumulator)
    print('Displaying lines.')
    show_image(hough, file, 'hough_out')