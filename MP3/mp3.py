import cv2
import numpy as np
import matplotlib.pyplot as plt

### INPUTS ###
file = 'moon.bmp'



### FUNCTIONS ###
def HistoEqualization(img, show_hist=False):
    # Create empty output image and histogram bins
    img_out = np.zeros(img.shape, dtype=np.uint8)
    hist_data = []
    hist_bins = [0] * 256

    # Iterate through pixels in input image, count quantity of each pixel value in hist bins
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            hist_bins[img[y, x]] += 1
            hist_data.append(img[y, x])

    # Iterate through hist bins and sum values to create CDF
    cdf = [0] * 256
    for i in range(255):
        cdf[i+1] = hist_bins[i+1] + cdf[i]

    # Iterate through pixels in input image, scale value, and out to output image
    hist_out = []
    for y in range(img_out.shape[0]):
        for x in range(img_out.shape[1]):
            val = int(round(cdf[img[y, x]] / max(cdf) * 255))
            img_out[y, x] = val
            hist_out.append(val)

    if show_hist:
        hist_comp(hist_data, hist_out)

    return img_out


def hist_comp(input_data, output_data):
    # Create a figure and two subplots, side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5)) 

    # Create the input image histogram
    ax1.hist(input_data, bins=256, color='blue')
    ax1.set_title('Input Image Histogram')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Count')

    # Create the output image histogram
    ax2.hist(output_data, bins=256, color='blue')
    ax2.set_title('Output Image Histogram')
    ax2.set_xlabel('Pixel Value')

    plt.show()


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

image_out = HistoEqualization(image, show_hist=True)
show_image(image_out, file, '_out')
