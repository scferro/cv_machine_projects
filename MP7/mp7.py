import numpy as np
import cv2
import matplotlib.pyplot as plt


### INPUTS ###
directory = 'image_girl'
total_images = 500  
output_path = 'output_video_ssd.mp4'



### FUNCTIONS ###

def ssd(template, image):
    return np.sum((template - image) ** 2)

def cross_correlation(template, image):
    return np.sum(template * image)

def normalized_cross_correlation(template, image):
    template_mean = template - np.mean(template)
    image_mean = image - np.mean(image)

    numerator = np.sum(template_mean * image_mean)
    denominator = np.sqrt(np.sum(template_mean ** 2) * np.sum(image_mean ** 2))

    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def track_face(template, image, search_window, method='ssd'):
    if method == 'ssd':
        comparison_func = ssd
    elif method == 'cross_correlation':
        comparison_func = cross_correlation
    elif method == 'normalized_cross_correlation':
        comparison_func = normalized_cross_correlation
    else:
        raise ValueError("Unknown method.")

    min_val = float('inf') if method == 'ssd' else -float('inf')
    best_position = (0, 0)
    t_rows, t_cols = template.shape[:2]
    for y in range(search_window[1], search_window[3] - t_rows):
        for x in range(search_window[0], search_window[2] - t_cols):
            sub_image = image[y:y + t_rows, x:x + t_cols]
            current_val = comparison_func(template, sub_image)
            if (method == 'ssd' and current_val < min_val) or (method in ['cross_correlation', 'normalized_cross_correlation'] and current_val > min_val):
                min_val = current_val
                best_position = (x, y)
    return best_position, min_val
    


### MAIN CODE ###

# Generate image paths 
image_paths = [f"{directory}/{i:04d}.jpg" for i in range(1, total_images + 1)]

# Load images
images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]

# Initialize tracking (assuming the face is roughly centered and occupies a known proportion of the frame)
initial_position = (54, 30)  # Top-left corner of the bounding box
box_width, box_height = 36, 36  # Width and height of the bounding box
template = images[0][initial_position[1]:initial_position[1]+box_height, initial_position[0]:initial_position[0]+box_width]

# Define the search window dimensions (a bit larger than the initial box to allow for movement)
search_window_margin = 20
search_windows = [
    (
        max(0, initial_position[0] - search_window_margin),
        max(0, initial_position[1] - search_window_margin),
        min(images[0].shape[1], initial_position[0] + box_width + search_window_margin),
        min(images[0].shape[0], initial_position[1] + box_height + search_window_margin)
    )
]

# Track the face in subsequent images
positions = [initial_position]
for i in range(1, len(images)):
    position, _ = track_face(template, images[i], search_windows[-1])
    positions.append(position)
    # Update the search window for the next frame
    search_windows.append(
        (
            max(0, position[0] - search_window_margin),
            max(0, position[1] - search_window_margin),
            min(images[0].shape[1], position[0] + box_width + search_window_margin),
            min(images[0].shape[0], position[1] + box_height + search_window_margin)
        )
    )

# Define the output video parameters
frame_size = (images[0].shape[1], images[0].shape[0])  # Width x Height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
video_writer = cv2.VideoWriter(output_path, fourcc, 15, frame_size)

# Draw bounding boxes on each image and add to the video
for img, pos in zip(images, positions):
    top_left = pos
    bottom_right = (pos[0] + box_width, pos[1] + box_height)
    # Draw the rectangle on the image
    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)  
    # Write the frame to the video
    video_writer.write(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
video_writer.release()

print(f"Video saved as {output_path}")
