import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load images and masks
def load_data(image_dir, mask_dir):
    images = []
    masks = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):  # Assuming images are in PNG format
            img = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
            images.append(img)
            masks.append(mask)
    
    return np.array(images), np.array(masks)

# Example usage
image_dir = 'path/to/images'
mask_dir = 'path/to/masks'
X, y = load_data(image_dir, mask_dir)

# Normalize images
X = X.astype('float32') / 255.0
y = (y > 0).astype('float32')  # Binarize masks