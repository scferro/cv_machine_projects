import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

### INPUT VALUES ###
image_files = ['test_ferro.png', 'face.bmp', 'face_old.bmp', 'gun.bmp', 'test.bmp']
size_filter_thresh = 100



### FUNCTIONS ###
def ccl(image):
    """
    The main CCL function
    """
    label_array = np.zeros([image.shape[0], image.shape[1]])
    num_labels = 0

    # Iterate through pixels and check values
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Access pixel value at (x, y)
            pixel = image[y, x]
            pixel = max(pixel)
            if pixel != 0:
                # Check if any neighbor pixels have been labelled
                neighbor_regions = check_neighbors(x, y, label_array)
                if len(neighbor_regions) == 1:
                    # If only on neighbor region, pixel is a part of that region
                    label_array[y, x] = neighbor_regions[0]
                elif len(neighbor_regions) == 0:
                    # If no neighbor regions, create new region
                    num_labels += 1
                    label_array[y, x] = num_labels
                elif len(neighbor_regions) > 1:
                    # If multiple neighbor regions, combine into one region
                    new_label = min(neighbor_regions)
                    # Iterate through each neighbor label
                    for label in neighbor_regions:
                        # If the neighbor label is not the min neighbor label, change it
                        if label != new_label:
                            num_labels += -1
                            label_array = change_labels(label_array, label, new_label)
                            if label <= num_labels:
                                label_array = change_labels(label_array, num_labels+1, label)

    label_array, num_labels = size_filter(label_array, num_labels)

    return label_array, num_labels


def size_filter(label_array, num_labels):
    """
    A function filter out small regions
    """
    label_count_list = [0] * num_labels

    # Iterate through neighbors and check values
    for y in range(label_array.shape[0]):
        for x in range(label_array.shape[1]):
            # Count number of each label
            label = label_array[y, x]
            if label != 0: 
                label_count_list[int(label)-1] += 1

    # If count of any label is lower than thresh, remove it 
    removed_labels = []
    for i in range(len(label_count_list)):
        target_label = i + 1
        if label_count_list[i] < size_filter_thresh:
            label_array = change_labels(label_array, target_label, 0)
            removed_labels.append(target_label)

    removed_labels.reverse()
    for label in removed_labels:
        for i in range(int(label), num_labels+1):
            print(i)
            change_labels(label_array, i, i-1)
        num_labels += -1


    return label_array, num_labels


def check_neighbors(x, y, label_array):
    """
    A function to check the labels of neighboring pixels
    """
    neighbor_regions = []
    x_list = [x-1, x, x+1]
    y_list = [y-1, y, y+1]
    labelled_pixel_count = 0

    # Iterate through neighbors and check values
    for y in y_list:
        for x in x_list:
            value = label_array[y, x]
            # If neighbor region exists and has not been seen before
            if value != 0: 
                labelled_pixel_count += 1
                if value not in neighbor_regions:
                    neighbor_regions.append(value)

    return neighbor_regions


def change_labels(label_array, old_label, new_label):
    """
    A function to change the a given label to a new value in a label array
    """
    new_label_array = label_array

    # Iterate through pixels in label array
    for y in range(label_array.shape[0]):
        for x in range(label_array.shape[1]):
            # If a pixel has the old label, change it to the new label in the new_label_array
            if label_array[y, x] == old_label:
                new_label_array[y, x] = new_label

    return new_label_array


def plot_labels(label_array, num_labels, file):
    """
    A function to plot the image with a different color in every region
    """
    color_list = []
    output_image = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
    print("Generating labelled image...")

    # Create random colors for each region
    for i in range(num_labels):
        color_list.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    # Iterate through each pixel in the label image and assign a color to the output image based on the labels
    for y in range(label_array.shape[0]):
        for x in range(label_array.shape[1]):
            label = label_array[y, x]
            if label != 0:
                # print(label)
                output_image[y, x] = color_list[int(label)-1]

    cv2.imshow('Color Image', output_image)
    
    # Wait for any key to be pressed
    print('Displaying image. Press any key to continue...')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    file_out = file[0:-4] + '_out.png'
    cv2.imwrite(file_out, output_image)



### MAIN CODE ###
for file in image_files:
    image = cv2.imread(file)
    label_array, num_labels = ccl(image)
    print('%d regions in image %s' % (num_labels, file))
    plot_labels(label_array, num_labels, file)