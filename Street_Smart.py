import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Path to your dataset directory (replace with actual path)
dataset_dir = r'C:\Users\xiaof\OneDrive\Desktop\CodeTheChange\Street_Smarts\archive\popular_street_foods\dataset'

# Parameters
img_height, img_width = 224, 224
batch_size = 32
validation_split = 0.2

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=validation_split
)

# Only rescale validation data
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split
)

# Training generator
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation generator
validation_generator = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Classes found:", train_generator.class_indices)