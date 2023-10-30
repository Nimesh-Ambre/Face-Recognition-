import cv2
import os

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video file
video_capture = cv2.VideoCapture('video.mp4')

# Create a directory to save the extracted faces
if not os.path.exists('extracted_faces'):
    os.mkdir('extracted_faces')

person_id = 0
frame_count = 0
target_frames = 5  # Number of frames to capture per person

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the frame to grayscale (for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # If a face is detected, extract and save it
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            detected_face = frame[y:y+h, x:x+w]

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add the label "Person id" below the rectangle
            label = "Person id"
            cv2.putText(frame, label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Save the detected face with the label
            face_file = f'extracted_faces/person_{person_id}_face_{frame_count}.jpg'
            cv2.imwrite(face_file, detected_face)
            frame_count += 1

        # Check if we have captured the maximum allowed frames per person
        if frame_count >= target_frames:
            frame_count = 0
            person_id += 1

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
