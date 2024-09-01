import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define path to your dataset directory
dataset_dir = 'dataset/train'

# Image data generator with validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% of data will be used for validation
)

# Training data generator
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Specify this is the training subset
)

# Validation data generator
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Specify this is the validation subset
)
