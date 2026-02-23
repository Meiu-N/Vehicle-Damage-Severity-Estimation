import tensorflow as tf                           
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  
from tensorflow.keras.applications import MobileNetV2   
from tensorflow.keras.optimizers import  Adam 

# Load a pre-trained MobileNetV2 model + top 
# layer, pre-trained on ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Add custom layers for damage severity estimation
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(3, activation ='softmax')  # Assuming 3 classes: 'No Damage', 'Minor Damage', 'Severe Damage'
])


# Freeze the base model layers (optional)
base_model.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up data generator for training (assuming images are in directories categorized by labels)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    'C:/data3a',  # Your dataset path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
model.fit(train_generator, epochs=10)
import cv2
import numpy as np

# Start camera feed
cap = cv2.VideoCapture(0)  # Change to 1 if you're using an external camera

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Resize frame to fit model input (224x224)
    img = cv2.resize(frame, (224, 224))

    # Convert image to a format suitable for prediction (expand dimensions to match input)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Shape should be (1, 224, 224, 3)

    # Predict severity of damage
    predictions = model.predict(img_array)
    
    # Get the class with the highest probability
    class_idx = np.argmax(predictions)
    class_labels = ['No Damage', 'Minor Damage', 'Severe Damage']
    severity = class_labels[class_idx]

    # Display the result on the frame
    cv2.putText(frame, f"Severity: {severity}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image
    cv2.imshow('Vehicle Damage Severity Estimation', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()