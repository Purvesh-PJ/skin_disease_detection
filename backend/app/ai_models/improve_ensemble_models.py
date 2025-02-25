import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Load pre-trained models
print("Loading saved models...")
resnet_model = load_model("../models/ResNet50_model.h5")
efficientnet_model = load_model("../models/EfficientNetB0_model.h5")
densenet_model = load_model("../models/DenseNet121_model.h5")

# Unfreeze last few layers for deeper fine-tuning
def unfreeze_layers(model, num_layers=50):
    for layer in model.layers[-num_layers:]:
        layer.trainable = True
    return model

resnet_model = unfreeze_layers(resnet_model)
efficientnet_model = unfreeze_layers(efficientnet_model)
densenet_model = unfreeze_layers(densenet_model)

# Lower learning rate for stability
learning_rate = 1e-5
optimizer = Adam(learning_rate=learning_rate)

resnet_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
efficientnet_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
densenet_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation for additional fine-tuning (no full preprocessing needed)
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()  # No augmentation for validation

# Load images (Assuming dataset is already structured properly)
train_generator = train_datagen.flow_from_directory("../dataset/train", target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_generator = val_datagen.flow_from_directory("../dataset/val", target_size=(224, 224), batch_size=32, class_mode='categorical')

# Fine-tune models without retraining from scratch
print("Fine-tuning models...")
resnet_model.fit(train_generator, epochs=5, validation_data=validation_generator)
efficientnet_model.fit(train_generator, epochs=5, validation_data=validation_generator)
densenet_model.fit(train_generator, epochs=5, validation_data=validation_generator)

# Save improved models
print("Saving updated models...")
resnet_model.save("../models/ResNet50_finetuned.h5")
efficientnet_model.save("../models/EfficientNetB0_finetuned.h5")
densenet_model.save("../models/DenseNet121_finetuned.h5")

print("Fine-tuning complete! ✅")
