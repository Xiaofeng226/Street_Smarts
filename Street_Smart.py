import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# Path to dataset (adjust this to your local machine)
# Require to change the directory for htat that downlaod it 
dataset_path = r'C:\Users\xiaof\OneDrive\Desktop\CodeTheChange\Street_Smarts\archive\popular_street_foods\dataset'

# How the image is viewed
img_height = 224
img_width = 224
batch_size = 32
validation_split = 0.2

# Generating image
train_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=validation_split
)

val_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=validation_split
)

# Load data from folders
train_data = train_generator.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = val_generator.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(train_data.class_indices.keys())
num_classes = len(class_names)

# Use MobileNetV2 as a base model (pretrained on ImageNet)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Add custom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model
epochs = 10
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# Evaluate on validation set
val_loss, val_acc = model.evaluate(val_data)
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

# Plot accuracy/loss curves
# Used chatgpt to plot the curves(I have only used R to plot curves so this is pretty new to me)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Predict on validation data
val_data.reset()
predictions = model.predict(val_data)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

# Confusion matrix & classification report  
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n")
print(cm)

# Check image count per class (for balance)
class_counts = np.bincount(train_data.classes)

plt.figure(figsize=(12, 6))
bars = plt.bar(class_names, class_counts)
plt.xticks(rotation=45, ha='right')
plt.title("Training Image Count per Class")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, str(yval), ha='center', va='bottom')

plt.tight_layout()
plt.show()