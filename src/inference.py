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
images, masks = load_data(image_dir, mask_dir)

# Normalize images
images = images.astype('float32') / 255.0
masks = masks.astype('float32') / 255.0

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)