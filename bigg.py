import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paths to your training and validation datasets (update with actual paths)
train_data_path = r"C:\data3a\training"
val_data_path = r"C:\data3a\validation"

# Function to load the data from directories
def load_data(data_path):
    X = []
    y = []
    for severity in os.listdir(data_path):
        severity_path = os.path.join(data_path, severity)
        if os.path.isdir(severity_path):
            for img_name in os.listdir(severity_path):
                img_path = os.path.join(severity_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (48, 48))  # Resize to a fixed size (48x48)
                    X.append(img_resized.flatten())  # Flatten the image
                    y.append(severity)  # Store the corresponding label (severity)
    return np.array(X), np.array(y)

# Load training and validation datasets
X_train, y_train = load_data(train_data_path)
X_val, y_val = load_data(val_data_path)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Validate the model
y_pred = knn.predict(X_val)
print("Classification Report for Validation Set:")
print(classification_report(y_val, y_pred))
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
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resized_frame = cv2.resize(gray_frame, (48, 48)).flatten().reshape(1, -1)  # Resize and flatten for KNN

        # Predict the damage severity using KNN
        if knn:
            severity_prediction = knn.predict(resized_frame)
            severity_label = severity_prediction[0]

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