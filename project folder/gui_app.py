import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained classifier model
model = load_model('retina_disease_model.h5')
categories = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# Image preprocessing
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128)) / 255.0
    return np.expand_dims(img, axis=0)

# GUI class
class RetinaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Retina Disease Detection")
        self.root.geometry("600x500")
        self.root.configure(bg="white")

        self.label = tk.Label(root, text="Upload a Retina Image", font=("Arial", 18), bg="white")
        self.label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=256, height=256, bg="lightgray")
        self.canvas.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 16), bg="white")
        self.result_label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image, font=("Arial", 14))
        self.upload_button.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path).resize((256, 256))
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

            # Predict
            input_img = preprocess_image(file_path)
            prediction = model.predict(input_img)
            result = categories[np.argmax(prediction)]
            self.result_label.config(text=f"Prediction: {result}")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = RetinaApp(root)
    root.mainloop()
