import cv2
import os
import numpy as np

# Path to Haar Cascade XML file for face detection
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to read and label face images
def get_images_and_labels(data_path):
    face_samples = []
    ids = []
    
    # Iterate through the images in the dataset directory
    for image_name in os.listdir(data_path):
        # Path to each image
        img_path = os.path.join(data_path, image_name)

        # Convert image to grayscale
        image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Detect faces
        faces = haar_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Extract the label from the image filename (assuming format: label.imagename.jpg)
        label = int(image_name.split('.')[0])

        # Store face samples and labels
        for (x, y, w, h) in faces:
            face_samples.append(image_gray[y:y + h, x:x + w])
            ids.append(label)

    return face_samples, ids

# Provide a path to your face images
data_path = 'dataset'  # Directory with training face images
faces, ids = get_images_and_labels(data_path)

# Train the recognizer with faces and their corresponding IDs
recognizer.train(faces, np.array(ids))

# Save the trained model to disk
recognizer.write('face_trained_model.yml')
print("Training completed and model saved!")
