import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the model
model = load_model('hand_sign_model.h5')

# Load class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def preprocess_image(img):
    img = cv2.resize(img, (100, 100))
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)
    return img

def predict_hand_sign(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    class_index = np.argmax(prediction)
    return class_labels[class_index]

def get_bounding_box(hand_landmarks, img_w, img_h):
    x_coords = [lm.x * img_w for lm in hand_landmarks]
    y_coords = [lm.y * img_h for lm in hand_landmarks]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    return x_min, y_min, x_max, y_max

# Streamlit app layout
st.title("Hand Sign Detection")
st.write("Press 'Start' to begin the webcam feed and detect hand signs.")

start = st.button('Start')
stop = st.button('Stop')

if start:
    # Set up webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        result = hands.process(rgb_frame)

        # If hands are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get bounding box for the hand
                img_h, img_w, _ = frame.shape
                x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks.landmark, img_w, img_h)

                # Crop the hand region
                hand_image = frame[y_min:y_max, x_min:x_max]

                # Predict hand sign
                sign = predict_hand_sign(hand_image)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the result
                cv2.putText(frame, f'Sign: {sign}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            # If no hand is detected, display a message
            cv2.putText(frame, 'No Hand Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the frame in Streamlit
        st.image(frame, channels="BGR")

        if stop:
            break

    cap.release()
    cv2.destroyAllWindows()
