from tensorflow.keras.utils import img_to_array
import cv2
import imutils
from keras.models import load_model
import numpy as np

face_cascade = cv2.CascadeClassifier(r'haarcascade_files\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'haarcascade_files\haarcascade_eye.xml')

video_file_name = r"sample\stanford_lecture2.mp4"
video_emotion_model_path = r'models\model_num.hdf5'

use_live_video = False
emotion_classifier = load_model(video_emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

cv2.namedWindow('Student Attention Detector')
cv2.namedWindow('Face Emotion Probabilities using AI')

cap = cv2.VideoCapture(0 if use_live_video else video_file_name)
if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error.")
            break

        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        canvas = np.zeros((350, 400, 3), dtype="uint8")
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi)

            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            roi = cv2.resize(roi, (48, 48)).astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[np.argmax(preds)]

            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                prob_percentage = prob * 100
                text = f"{emotion}: {prob_percentage:.2f}%"
                bar_width = int(prob * 300)
                color = (0, 255, 0) if emotion == label else (0, 0, 255)
                cv2.rectangle(canvas, (10, i * 35 + 5), (10 + bar_width, i * 35 + 30), color, -1)
                cv2.putText(canvas, text, (10, i * 35 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Student Attention Detector', frame)
        cv2.imshow('Face Emotion Probabilities using AI', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break
    except Exception as e:
        print(f"Error: {e}")
        break

cap.release()
cv2.destroyAllWindows()
