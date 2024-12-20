import cv2
import numpy as np
from datetime import datetime
import os
import csv
from deepface import DeepFace

# Initialize logging file
log_file = "face_detection_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Face ID", "Detection Time", "Bounding Box", "Age", "Gender", "Image File", "Identity"])

# Directory to save face images
faces_dir = "detected_faces"
os.makedirs(faces_dir, exist_ok=True)

# Directory where known faces are stored
known_faces_dir = "known_faces"
if not os.path.exists(known_faces_dir):
    print(f"Error: Directory '{known_faces_dir}' not found. Please create it and add images of known faces.")
    exit(1)  # Exit if the known_faces directory doesn't exist

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not available.")
    exit()

print("Camera initialized and working.")

# Initialize variables
face_id_counter = 0
faces_info = {}  # Tracks detected faces: {face_id: {bbox, embedding, last_seen}}
known_faces = {}  # Stores embeddings of known faces {name: embedding}
similarity_threshold = 30  # Threshold for embedding similarity (Euclidean distance)
max_disappeared_frames = 10  # Maximum frames a face can disappear before being removed

# Function to compute face embedding using DeepFace
def get_face_embedding(face_image):
    try:
        embedding = DeepFace.represent(face_image, model_name='Facenet', enforce_detection=False)
        return np.array(embedding[0]["embedding"])
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Function to load pre-enrolled known faces
def load_known_faces(known_faces_dir):
    known_faces = {}
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Handle both jpg and png images
            name = filename.split("_")[0]  # Assuming the format is Name_date.jpg
            img_path = os.path.join(known_faces_dir, filename)
            face = cv2.imread(img_path)
            embedding = get_face_embedding(face)
            if embedding is not None:
                known_faces[name] = embedding
            else:
                print(f"Skipping image {filename} due to error in embedding.")
    return known_faces

# Load pre-enrolled known faces
known_faces = load_known_faces(known_faces_dir)

# Function to match a new face with existing faces
def find_matching_face(embedding, faces_info, threshold):
    for face_id, info in faces_info.items():
        if np.linalg.norm(embedding - info["embedding"]) < threshold:
            return face_id
    return None

# Function to recognize the face from known faces
def recognize_known_face(embedding, known_faces, threshold):
    for name, known_embedding in known_faces.items():
        if np.linalg.norm(embedding - known_embedding) < threshold:
            return name
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera feed not available.")
        break

    # Convert to grayscale and apply histogram equalization
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(equalized, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
    print(f"Faces detected: {len(faces)}")

    current_face_ids = set()

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]  # Extract region of interest (face)
        embedding = get_face_embedding(face_roi)  # Compute face embedding

        if embedding is None:
            print("Embedding not generated for this face.")
            continue

        # Check if the face matches any known face
        identity = recognize_known_face(embedding, known_faces, similarity_threshold)

        if identity:  # Known face detected
            print(f"Known face detected: {identity}")
            face_id = identity  # Assign the name of the recognized person
        else:  # New face detected
            face_id_counter += 1
            face_id = face_id_counter
            faces_info[face_id] = {
                "bbox": (x, y, w, h),
                "embedding": embedding,
                "last_seen": 0  # Reset disappearance count
            }

            # Predict age and gender using DeepFace
            try:
                print("Analyzing face with DeepFace...")
                analysis = DeepFace.analyze(face_roi, actions=['age', 'gender'], enforce_detection=False)
                age = analysis[0]['age']
                gender = analysis[0]['gender']
                print(f"Age: {age}, Gender: {gender}")
            except Exception as e:
                age = "Unknown"
                gender = "Unknown"
                print(f"Error predicting age/gender: {e}")

            # Save the detected face image
            image_filename = f"{faces_dir}/face_{face_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(image_filename, face_roi)
            print(f"Face image saved as: {image_filename}")

            # Log detection
            detection_time = datetime.now().strftime('%H:%M:%S')
            with open(log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([face_id, detection_time, (x, y, w, h), age, gender, image_filename, identity])

        current_face_ids.add(face_id)

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"Face ID: {face_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Update disappearance counts for unmatched faces
    for face_id in list(faces_info.keys()):
        if face_id not in current_face_ids:
            faces_info[face_id]["last_seen"] += 1
            if faces_info[face_id]["last_seen"] > max_disappeared_frames:
                del faces_info[face_id]  # Remove disappeared face

    # Display total unique faces detected
    unique_faces_count = len(faces_info)
    cv2.putText(frame, f"Total Unique Faces: {unique_faces_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the video feed
    cv2.imshow('Enhanced Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
