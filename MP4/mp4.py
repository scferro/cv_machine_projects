import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

### INPUTS ###
image_paths = ['gun.bmp', 'joy.bmp', 'pointer.bmp']
skin_pixels = []
selected_coordinates = []
thresh = 20
kernel_3 = np.ones((5,5), np.uint8)
kernel_5 = np.ones((5,5), np.uint8)
kernel_7 = np.ones((7,7), np.uint8)


### FUNCTIONS ###
def load_and_show_image(image_path):
    # Loads and displays image, converts to HSV and returns
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.show()
    return image_hsv

def onselect(eclick, erelease):
    # Function for selecting regions
    x1, y1, x2, y2 = int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)
    selected_coordinates.append((x1, y1, x2, y2))
    print(f"Region selected from ({x1}, {y1}) to ({x2}, {y2})")

def interactive_select_skin_pixels(image_path):
    # Function for selecting skin pixels from photos
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    rect_selector = RectangleSelector(ax, onselect, drawtype='box',
                                      useblit=True, button=[1],
                                      minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    plt.show()
    return image_hsv, selected_coordinates

def get_skin_pixels(image, coordinates):
    # Finds skin pixels in image based on trained model
    skin_pixels = []
    for (x1, y1, x2, y2) in coordinates:
        region = image[y1:y2, x1:x2]
        for pixel in region.reshape(-1, 3):
            skin_pixels.append(pixel) 
    return np.array(skin_pixels)

def initialize_centroids(data, k):
    # Created initial centroids for K means
    indices = np.random.choice(np.arange(len(data)), size=k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    # Assigns data to clusters for K means
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, labels, k):
    # Updates centroids for K means
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def k_means(data, k, iterations=100):
    # Trains K means model
    centroids = initialize_centroids(data, k)
    for i in range(iterations):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

def create_skin_mask(test_pixels, centroids, threshold=thresh):
    # Returns a skin mask of an image
    distances = np.sqrt(((test_pixels - centroids[:, np.newaxis])**2).sum(axis=2))
    closest_distances = np.min(distances, axis=0)
    return (closest_distances < threshold).astype(int)



### MAIN CODE ###
# Select skin regions from photos
for training_image_path in image_paths:
    training_image_hsv, skin_coordinates = interactive_select_skin_pixels(training_image_path)
    pixels = get_skin_pixels(training_image_hsv, skin_coordinates)
    if skin_pixels == []:
        skin_pixels = pixels
    else:
        skin_pixels = np.vstack((skin_pixels, pixels))
    selected_coordinates = []

# Train the custom K-Means model
centroids, labels = k_means(skin_pixels, 10)

# Process each test image
for test_image_path in image_paths:
    test_image_hsv = load_and_show_image(test_image_path)
    test_pixels = test_image_hsv.reshape(-1, 3)

    # Create skin mask based on distance to nearest cluster centroid
    skin_mask = create_skin_mask(test_pixels, centroids).reshape(test_image_hsv.shape[:2])
    skin_mask = skin_mask.astype(np.uint8)

    # Display the detected skin regions
    plt.imshow(skin_mask, cmap='gray')
    plt.title('Detected Skin Regions')
    plt.show()

    skin_mask_out = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_3)
    skin_mask_out = cv2.morphologyEx(skin_mask_out, cv2.MORPH_OPEN, kernel_3)
    skin_mask_out = cv2.morphologyEx(skin_mask_out, cv2.MORPH_CLOSE, kernel_5)

    # Display the detected skin regions
    plt.imshow(skin_mask_out, cmap='gray')
    plt.title('Detected Skin Regions')
    plt.show()