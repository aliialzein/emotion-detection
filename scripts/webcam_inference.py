import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your newly trained model
model = load_model('models/emotion_model.keras')

# Emotion labels (same order used in training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection (but keep RGB frame for model)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract color face region
        face = frame[y:y+h, x:x+w]

        # Resize to 96x96 as expected by MobileNetV2
        face = cv2.resize(face, (96, 96))

        # Preprocessing
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Prediction
        prediction = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
