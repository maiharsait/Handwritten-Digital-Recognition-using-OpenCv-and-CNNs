import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import io
import warnings

warnings.filterwarnings('ignore')


# ===============================================================
#  ADVANCED CNN MODEL CLASS
# ===============================================================
class AdvancedDigitRecognitionModel:

    def __init__(self, model_path='digit_model.h5'):
        self.model_path = model_path
        self.model = None
        self.is_trained = False

    def build_model(self):
        print("[*] Building Advanced CNN Model...")

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),

            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),

            layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("[✓] Model Built Successfully")
        return model

    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=128):
        if self.model is None:
            self.build_model()

        print("[*] Training Model with Data Augmentation...")

        data_augmentation = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )

        history = self.model.fit(
            data_augmentation.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1,
            steps_per_epoch=len(X_train) // batch_size
        )

        self.is_trained = True
        print("[✓] Model Training Completed")
        return history

    def save_model(self):
        if self.model is not None:
            self.model.save(self.model_path)
            print(f"[✓] Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            self.is_trained = True
            print(f"[✓] Model loaded from {self.model_path}")
            return True
        return False

    def predict(self, image_array):
        if self.model is None:
            return None, None

        image_array = np.expand_dims(image_array, axis=-1)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = self.model.predict(image_array, verbose=0)
        digit = np.argmax(predictions[0])
        confidence = float(predictions[0][digit]) * 100

        return digit, confidence


# ===============================================================
#  DATA PREPROCESSOR
# ===============================================================
class DataProcessor:
    @staticmethod
    def load_and_preprocess_data():
        print("[*] Loading MNIST Dataset...")

        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        print("[✓] MNIST Preprocessing Complete")

        return X_train, y_train, X_test, y_test


# ===============================================================
#  DRAWN IMAGE PREPROCESSOR – FIXED VERSION
# ===============================================================
class CanvasImageProcessor:

    @staticmethod
    def preprocess_drawn_digit(image):
        if image is None:
            return None

        img = image.convert('L')
        img = np.array(img)

        # invert (white bg -> 0, black digit -> 255)
        img = 255 - img

        # threshold
        _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

        # find bounding box
        coords = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(coords)

        # crop
        img = img[y:y+h, x:x+w]

        # resize to 20x20
        img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)

        # create 28x28 canvas
        canvas = np.zeros((28, 28), dtype=np.float32)
        offset_x = (28 - 20) // 2
        offset_y = (28 - 20) // 2
        canvas[offset_y:offset_y + 20, offset_x:offset_x + 20] = img

        # normalize
        canvas /= 255.0

        return canvas


# ===============================================================
#  GUI FOR DIGIT RECOGNITION
# ===============================================================
class DigitRecognitionGUI(tk.Tk):

    def __init__(self, model):
        super().__init__()

        self.model = model

        self.title("Handwritten Digit Recognition - CNN")
        self.geometry("900x700")
        self.resizable(False, False)

        self.canvas_width = 280
        self.canvas_height = 280
        self.last_x = 0
        self.last_y = 0
        self.drawing = False

        self.setup_ui()

    def setup_ui(self):
        title = tk.Label(
            self, text="Handwritten Digit Recognition using CNN",
            font=("Arial", 20, "bold")
        )
        title.pack(pady=15)

        frame = tk.Frame(self)
        frame.pack(pady=10)

        # drawing canvas
        self.canvas = tk.Canvas(
            frame, width=self.canvas_width, height=self.canvas_height,
            bg="white", bd=2, relief=tk.SUNKEN, cursor="cross"
        )
        self.canvas.pack(side=tk.LEFT, padx=20)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        right_frame = tk.Frame(frame)
        right_frame.pack(side=tk.RIGHT)

        self.result_digit = tk.Label(
            right_frame, text="?",
            font=("Arial", 80, "bold"), fg="blue"
        )
        self.result_digit.pack(pady=20)

        self.confidence_label = tk.Label(
            right_frame, text="Confidence: 0%",
            font=("Arial", 14)
        )
        self.confidence_label.pack(pady=10)

        tk.Button(
            right_frame, text="Recognize", font=("Arial", 12, "bold"),
            bg="#27ae60", fg="white", padx=15, pady=10,
            command=self.recognize_digit
        ).pack(pady=10)

        tk.Button(
            right_frame, text="Clear", font=("Arial", 12, "bold"),
            bg="#e74c3c", fg="white", padx=15, pady=10,
            command=self.clear_canvas
        ).pack(pady=10)

    # Draw controls
    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self.drawing = True

    def draw(self, event):
        if self.drawing:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill="black", width=20, capstyle=tk.ROUND, smooth=True
            )
            self.last_x = event.x
            self.last_y = event.y

    def stop_draw(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_digit.config(text="?")
        self.confidence_label.config(text="Confidence: 0%")

    # Capture and recognize
    def get_canvas_image(self):
        img = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        draw = ImageDraw.Draw(img)

        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) >= 4:
                for i in range(0, len(coords)-2, 2):
                    draw.line(
                        (coords[i], coords[i+1], coords[i+2], coords[i+3]),
                        fill="black", width=20
                    )
        return img

    def recognize_digit(self):
        img = self.get_canvas_image()
        processed = CanvasImageProcessor.preprocess_drawn_digit(img)

        digit, conf = self.model.predict(processed)

        self.result_digit.config(text=str(digit))
        self.confidence_label.config(text=f"Confidence: {conf:.2f}%")

        print(f"Predicted: {digit} - {conf:.2f}%")



# ===============================================================
#  MAIN FUNCTION
# ===============================================================
def main():
    print("====================================")
    print("  HANDWRITTEN DIGIT RECOGNITION APP ")
    print("====================================")

    model = AdvancedDigitRecognitionModel()

    if model.load_model():
        print("[✓] Loaded pre-trained model.")
    else:
        print("[!] No saved model found. Training a new model...")
        X_train, y_train, X_test, y_test = DataProcessor.load_and_preprocess_data()
        model.build_model()
        model.train(X_train, y_train, X_test, y_test, epochs=20)
        model.save_model()

    print("[*] Launching GUI...")
    app = DigitRecognitionGUI(model)
    app.mainloop()


if __name__ == "__main__":
    main()
