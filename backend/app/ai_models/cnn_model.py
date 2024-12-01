from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from utils.preprocess import get_data_generators

def create_custom_cnn(input_shape=(600, 450, 3), num_classes=7):
    print("Initializing model...")
    model = Sequential()

    # Use Input layer for specifying the input shape
    model.add(Input(shape=input_shape))
    print(f"Input layer added with shape: {input_shape}")

    # First Convolutional Block
    model.add(Conv2D(32, (3, 3), activation=None))  # No activation yet
    model.add(BatchNormalization())  # BatchNormalization after Conv2D
    model.add(Activation('relu'))  # ReLU activation after BatchNormalization
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("First convolutional block added.")
    
    # Second Convolutional Block
    model.add(Conv2D(64, (3, 3), activation=None))  # No activation yet
    model.add(BatchNormalization())  # BatchNormalization after Conv2D
    model.add(Activation('relu'))  # ReLU activation after BatchNormalization
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("Second convolutional block added.")
    
    # Third Convolutional Block
    model.add(Conv2D(128, (3, 3), activation=None))  # No activation yet
    model.add(BatchNormalization())  # BatchNormalization after Conv2D
    model.add(Activation('relu'))  # ReLU activation after BatchNormalization
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("Third convolutional block added.")
    
    # Fourth Convolutional Block
    model.add(Conv2D(256, (3, 3), activation=None))  # No activation yet
    model.add(BatchNormalization())  # BatchNormalization after Conv2D
    model.add(Activation('relu'))  # ReLU activation after BatchNormalization
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("Fourth convolutional block added.")
    
    # Flatten and Fully Connected Layers
    model.add(Flatten())
    print("Flatten layer added.")
    
    # Add L2 regularization to the Dense layers
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # Apply L2 regularization
    model.add(Dropout(0.5))
    print("Dense layer with 256 units and L2 regularization added.")
    
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # Apply L2 regularization
    model.add(Dropout(0.5))
    print("Dense layer with 128 units and L2 regularization added.")
    
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    print(f"Output layer added for {num_classes} classes.")
    print("Model Initialization complete successfully")
    
    return model

# Create the model
print("-------------------------------")
print("CREATING CNN MODEL...")
print("-------------------------------")
model = create_custom_cnn()
print("Model created successfully.")
print("\n")

# Compile the model
print("-------------------------------")
print("COMPILING CNN MODEL...")
print("-------------------------------")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully.")
print("\n")

# Get the data generators for training and validation
print("-------------------------------")
print("GETTING DATA GENERATORS...")
print("-------------------------------")

train_generator, validation_generator, label_encoder = get_data_generators(sample_size=1000)

# Check if the generators are None and handle the error
if train_generator is None or validation_generator is None:
    print("Failed to obtain data generators. Exiting...")
    exit()  # Stop the program to avoid further errors
print("Data generators obtained successfully.")

# Define learning rate scheduler
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
)

# Train the model
print("Starting model training...")
model.fit(train_generator, epochs=30, validation_data=validation_generator, callbacks=[lr_scheduler, early_stopping ])
print("Model training completed.")

# Save the trained model
print("Saving the trained model...")
model.save(r'D:\skin_disease_detection\backend\app\static\trained_models\skin_disease_model.h5')
print("Model saved successfully.")

# Display the model architecture
print("Displaying model summary...")
model.summary()

print("-----------------------------------------------")
print("Evaluation model on validation data")
print("-----------------------------------------------")
print("Evaluating model on test data...")

# Assuming the 'validation_generator' contains the data you want to test on
test_generator = validation_generator  # Use validation data as test data

evaluation = model.evaluate(test_generator)
print(f"Test Accuracy: {evaluation[1]*100:.2f}%")