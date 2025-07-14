from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('retina_disease_model.h5')
categories = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    return categories[class_idx]

print(predict_image('test_image.jpg'))
