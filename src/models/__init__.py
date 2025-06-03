import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Load images and masks
def load_data(image_dir, mask_dir):
    images = []
    masks = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):  # Assuming images are in PNG format
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)  # Assuming masks have the same filename
            
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            images.append(image)
            masks.append(mask)
    
    images = np.array(images)
    masks = np.array(masks)
    
    # Normalize images
    images = images.astype('float32') / 255.0
    masks = masks.astype('float32') / 255.0
    
    # Reshape masks for binary classification
    masks = np.expand_dims(masks, axis=-1)
    
    return images, masks

# Example usage
image_dir = 'path/to/images'
mask_dir = 'path/to/masks'
X, y = load_data(image_dir, mask_dir)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)