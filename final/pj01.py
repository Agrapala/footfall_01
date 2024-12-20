import cv2
import numpy as np
from datetime import datetime
import os
import csv
from deepface import DeepFace  # For age and gender prediction

# Initialize logging file
log_file = "face_detection_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Face ID", "Detection Time", "Bounding Box", "Age", "Gender"])

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default camera
cap = cv2.VideoCapture(0)

# Initialize variables
face_id_counter = 0
faces_info = {}  # Stores Face ID, bbox, and embeddings

# Function to compute face embedding (mean color of face region)
def get_face_embedding(face_image):
    return np.mean(face_image, axis=(0, 1))

# Function to compare embeddings and match faces
def find_matching_face(embedding, faces_info, threshold=50):
    for face_id, info in faces_info.items():
        if np.linalg.norm(embedding - info["embedding"]) < threshold:  # Euclidean distance
            return face_id  # Found a match
    return None  # No match found

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera feed not available.")
        break

    # Convert to grayscale and apply histogram equalization
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(equalized, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]  # Extract region of interest (face)
        embedding = get_face_embedding(face_roi)  # Compute face embedding

        # Check if the face matches an existing one
        match = find_matching_face(embedding, faces_info)

        if match is None:  # New face detected
            face_id_counter += 1
            faces_info[face_id_counter] = {
                "bbox": (x, y, w, h),
                "embedding": embedding
            }

            # Predict age and gender using DeepFace
            try:
                analysis = DeepFace.analyze(face_roi, actions=['age', 'gender'], enforce_detection=False)
                age = analysis[0]['age']
                gender = analysis[0]['gender']
            except Exception as e:
                age = "Unknown"
                gender = "Unknown"
                print(f"Error predicting age/gender: {e}")

            # Log detection
            detection_time = datetime.now().strftime('%H:%M:%S')
            with open(log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([face_id_counter, detection_time, (x, y, w, h), age, gender])

        else:  # Existing face detected
            face_id = match

        # Draw bounding box and ID
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"Face ID: {face_id if match else face_id_counter}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display total unique faces detected
    cv2.putText(frame, f"Total Unique Faces: {len(faces_info)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the video feed
    cv2.imshow('Face Detection with Low-Light Enhancement', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
