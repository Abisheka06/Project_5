import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


IMG_SIZE = 224
DATASET_DIR = 'Solar_Panel_Dataset'
CATEGORIES = os.listdir(DATASET_DIR)

data = []
labels = []

for idx, category in enumerate(CATEGORIES):
    category_path = os.path.join(DATASET_DIR, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # Normalize
            data.append(img)
            labels.append(idx)
        except:
            continue

X = np.array(data)
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

print(f"Total images loaded: {len(X)}")
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

import seaborn as sns
import pandas as pd


label_map = {i: cat for i, cat in enumerate(CATEGORIES)}
y_named = [label_map[lbl] for lbl in y]


df = pd.DataFrame({'Label': y_named})
plt.figure(figsize=(8,6))
sns.countplot(x='Label', data=df)
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


def show_samples(X, y, label_map, num=3):
    classes = list(label_map.keys())
    plt.figure(figsize=(num * 3, len(classes) * 3))
    for idx, class_idx in enumerate(classes):
        class_indices = np.where(y == class_idx)[0]
        selected_indices = np.random.choice(class_indices, size=num, replace=False)
        for j, img_idx in enumerate(selected_indices):
            plt.subplot(len(classes), num, idx * num + j + 1)
            plt.imshow(X[img_idx])
            plt.axis('off')
            plt.title(label_map[class_idx])
    plt.tight_layout()
    plt.show()

show_samples(X, y, label_map, num=3)

import tensorflow as tf
from keras import layers, models

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax')  # 6 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=16,
                    validation_data=(X_test, y_test))

print("Saving model to:", os.getcwd())
model.save('solar_panel_model.h5')

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

import matplotlib.pyplot as plt

# Accuracy plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()



