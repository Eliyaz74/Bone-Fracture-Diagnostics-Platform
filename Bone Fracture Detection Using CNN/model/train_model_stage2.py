import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
from PIL import ImageFile

# ✅ Fix truncated/corrupted image errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Paths - adjust based on your dataset structure
train_dir = "../dataset/stage2_MultiClassification/train"
test_dir = "../dataset/stage2_MultiClassification/test"

# Data Generators - include basic augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',   # ✅ Multi-class classification
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Print class indices for mapping predictions later
print("Class Indices:", train_data.class_indices)

# Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # ✅ Add dropout to prevent overfitting
    layers.Dense(train_data.num_classes, activation='softmax')  # Multi-class output
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the Model
model.fit(
    train_data,
    validation_data=test_data,
    epochs=15,  # ✅ Increase epochs for better training
    verbose=1
)

# Ensure saved_model folder exists
save_path = os.path.join(os.getcwd(), "saved_model")
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save Stage 2 Model
model.save(os.path.join(save_path, "stage2_model.h5"))
print("✅ Stage 2 Model Saved Successfully!")