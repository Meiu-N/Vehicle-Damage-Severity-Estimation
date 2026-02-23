import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam

# Paths to your training and validation datasets (update with actual paths)
train_data_path = r"C:\data3a\training"
val_data_path = r"C:\data3a\validation"

# Define image size for ResNet50 input
IMG_SIZE = (224, 224)  # ResNet50 input size

# Load ResNet50 model, without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(len(os.listdir(train_data_path)), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators for training and validation
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
    train_data_path, target_size=IMG_SIZE, batch_size=32, class_mode='categorical', color_mode='rgb'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator = val_datagen.flow_from_directory(
    val_data_path, target_size=IMG_SIZE, batch_size=32, class_mode='categorical', color_mode='rgb'
)

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Evaluate model
val_generator.reset()
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())
print("Classification Report for Validation Set:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))

# Function to detect vehicle damage severity in webcam feed
def detect_damage_severity_in_webcam():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, IMG_SIZE)
        preprocessed_frame = preprocess_input(resized_frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension

        # Predict the damage severity
        severity_prediction = model.predict(preprocessed_frame)
        severity_label = class_labels[np.argmax(severity_prediction)]

        # Display the predicted damage severity on the frame
        cv2.putText(frame, f"Damage Severity: {severity_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Vehicle Damage Detection", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the damage severity detection on webcam feed
detect_damage_severity_in_webcam()

