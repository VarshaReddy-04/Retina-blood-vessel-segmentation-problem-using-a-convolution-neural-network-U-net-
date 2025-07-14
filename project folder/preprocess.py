import cv2
import os
import numpy as np

def load_images_from_folder(folder, size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, size)
            img = img / 255.0
            images.append(img)
    return np.array(images)
