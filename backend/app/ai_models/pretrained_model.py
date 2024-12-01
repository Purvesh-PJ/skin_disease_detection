from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from utils.preprocess import get_data_generators

from tensorflow.keras.layers import BatchNormalization

def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=7, trainable_layers=10):
    print("Initializing transfer learning model with ResNet50...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Fine-tune only the last few layers
    for layer in base_model.layers[:-4]:  # Freeze more layers
        layer.trainable = False

    for layer in base_model.layers[-4:]:  # Unfreeze the last 4 layers for fine-tuning
        layer.trainable = True


    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add custom dense layers with batch normalization and dropout
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)  # Batch normalization layer
    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)  # Batch normalization for new dense layer
    x = Dropout(0.3)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # Adjusted dropout rate

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    print("Model initialized successfully with added dense layer and batch normalization.")

    return model



# Create the model
print("-------------------------------")
print("CREATING TRANSFER LEARNING MODEL...")
print("-------------------------------")
model = create_transfer_learning_model()
print("Model created successfully.\n")

# Compile the model
print("-------------------------------")
print("COMPILING TRANSFER LEARNING MODEL...")
print("-------------------------------")
model.compile(optimizer=Adam(learning_rate=5e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully.\n")

# Get data generators for training and validation
print("-------------------------------")
print("GETTING DATA GENERATORS...")
print("-------------------------------")
train_generator, validation_generator, label_encoder = get_data_generators(sample_size=1000)

# Check if the generators are None and handle the error
if train_generator is None or validation_generator is None:
    print("Failed to obtain data generators. Exiting...")
    exit()  # Stop the program to avoid further errors
print("Data generators obtained successfully.")

# Define callbacks for learning rate and early stopping
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=0.5, patience=3, min_lr=1e-6, verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
)

# Train the model
print("Starting model training...")
model.fit(train_generator, epochs=30, validation_data=validation_generator, callbacks=[lr_scheduler, early_stopping])
print("Model training completed.")

# Save the trained model
print("Saving the trained model...")
model.save(r'D:\skin_disease_detection\backend\app\static\trained_models\skin_disease_model.h5')
print("Model saved successfully.")

# Display the model architecture
print("Displaying model summary...")
model.summary()

print("-----------------------------------------------")
print("Evaluating model on validation data")
print("-----------------------------------------------")
evaluation = model.evaluate(validation_generator)
print(f"Test Accuracy: {evaluation[1]*100:.2f}%")
