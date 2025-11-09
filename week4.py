# Same Kaggle setup, download, and unzip as before
!pip install kaggle -q
from google.colab import files
print("Upload your kaggle.json API key:")
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c iuss-23-24-automatic-diagnosis-breast-cancer
!unzip -q iuss-23-24-automatic-diagnosis-breast-cancer.zip -d dataset

# Importing libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# Data preprocessing
train_dir = "dataset/train"  # adjust if needed
test_dir = "dataset/test"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # smaller image size for faster training
    batch_size=16,         # smaller batch
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Simplified CNN model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compiling and training (fewer epochs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_data=val_generator, epochs=5)

# Ploting results
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# Prediction example
img_path = "dataset/test/some_image.jpg"  # change to an actual image path
img = image.load_img(img_path, target_size=(64, 64))
img_array = tf.expand_dims(image.img_to_array(img) / 255.0, 0)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
print("Predicted class:", predicted_class)