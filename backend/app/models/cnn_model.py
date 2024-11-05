from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from utils.test_preprocess import testing_data_generators

def create_custom_cnn(input_shape=(600, 450, 3), num_classes=7):
    print("Initializing model...")
    model = Sequential()
    
    # Use Input layer for specifying the input shape
    model.add(Input(shape=input_shape))
    print(f"Input layer added with shape: {input_shape}")

    # First Convolutional Block
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("First convolutional block added.")
    
    # Second Convolutional Block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("Second convolutional block added.")
    
    # Third Convolutional Block
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("Third convolutional block added.")
    
    # Fourth Convolutional Block
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("Fourth convolutional block added.")
    
    # Flatten and Fully Connected Layers
    model.add(Flatten())
    print("Flatten layer added.")
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    print("Dense layer with 256 units added.")
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    print("Dense layer with 128 units added.")
    
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    print(f"Output layer added for {num_classes} classes.")
    
    return model

# Create the model
print("Creating CNN model...")
model = create_custom_cnn()
print("Model created successfully.")

# Compile the model
print("Compiling model...")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Model compiled.")

# Get the data generators for training and validation
print("Getting data generators...")
train_generator, validation_generator = testing_data_generators()
print("Frm CNN Train Generator:", train_generator)
print("Frm CNN Validation Generator:", validation_generator)

# Check if the generators are None and handle the error
if train_generator is None or validation_generator is None:
    print("Failed to obtain data generators. Exiting...")
    exit()  # Stop the program to avoid further errors

print("Data generators obtained.")

# Train the model
print("Starting model training...")
model.fit(train_generator, epochs=3, validation_data=validation_generator)
print("Model training completed.")

# Save the trained model
print("Saving the trained model...")
model.save(r'D:\skin_disease_detection\backend\app\static\trained_models\skin_disease_model.h5')
print("Model saved successfully.")

# Display the model architecture
print("Displaying model summary...")
model.summary()
