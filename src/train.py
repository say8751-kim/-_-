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
            img = cv2.imread(os.path.join(image_dir, filename))
            img = cv2.resize(img, (128, 128))  # Resize to 128x128
            images.append(img)
            
            mask_filename = filename.replace('.png', '_mask.png')  # Assuming masks have a similar naming convention
            mask = cv2.imread(os.path.join(mask_dir, mask_filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (128, 128))
            masks.append(mask)
    
    return np.array(images), np.array(masks)

# Example usage
image_dir = 'path/to/images'
mask_dir = 'path/to/masks'
X, y = load_data(image_dir, mask_dir)

# Normalize images
X = X / 255.0
y = y / 255.0
y = np.expand_dims(y, axis=-1)  # Add channel dimension for masks

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)