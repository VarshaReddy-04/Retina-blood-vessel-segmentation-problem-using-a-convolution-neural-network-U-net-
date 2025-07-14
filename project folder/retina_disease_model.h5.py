from preprocess import load_images_from_folder
from disease_classifier import build_classifier
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

categories = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
X, y = [], []

for idx, category in enumerate(categories):
    imgs = load_images_from_folder(f'datasets/{category}')
    labels = [idx] * len(imgs)
    X.extend(imgs)
    y.extend(labels)

X = np.array(X)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = build_classifier()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save('retina_disease_model.h5')
