import cv2

# Load Haar Cascade for face detection
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the trained LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_trained_model.yml')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Create dictionary of names based on IDs
names = {1: "Lionel Messi", 2: "Cristiano Ronaldo", 3:"Narendra Modi"}  # Add more IDs and names

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale (since the recognizer works on grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over each face detected
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray_frame[y:y + h, x:x + w]

        # Use recognizer to predict the face ID and confidence level
        id, confidence = recognizer.predict(face_roi)

        # If confidence is low, it means the face was recognized
        if confidence < 100:
            name = names.get(id, "Unknown")
            confidence_text = f"  {100 - confidence:.2f}% Confidence"
        else:
            name = "Unknown"
            confidence_text = f"  {100 - confidence:.2f}% Confidence"

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display name and confidence on the image
        cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (x + 5, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
