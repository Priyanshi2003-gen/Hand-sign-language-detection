import cv2
import os

# Set up the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create directories for saving images
dataset_dir = 'dataset/train'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Class labels
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]  # Replace with your actual class names
current_class = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow('Capture Images', frame)
    
    # Capture images for the current class
    if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save image
        class_dir = os.path.join(dataset_dir, class_labels[current_class])
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        img_count = len(os.listdir(class_dir))
        if img_count >= 200:  # Adjust based on how many images you want per class
            print(f"Collected enough images for class {class_labels[current_class]}")
        else:
            img_name = f"img{img_count + 1}.jpg"
            img_path = os.path.join(class_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"Saved {img_name}")

    # Change class if 'n' is pressed
    if cv2.waitKey(1) & 0xFF == ord('n'):
        current_class = (current_class + 1) % len(class_labels)
        print(f"Switched to class {class_labels[current_class]}")
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
