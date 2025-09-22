import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import os
import csv
from PIL import Image


# Load the pre-trained face recognition model
def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image_path in path:
        try:
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            imageNP = np.array(img, 'uint8')
            
            # Assuming filenames are in the format 'user<ID>.jpg', or 'user.<ID>.<number>.jpg'
            filename = os.path.split(image_path)[1]  # Extract filename

            # Extract ID using a more flexible approach
            id_str = filename.split(".")[1]  # Get the second part after 'user'
            id = int(id_str)  # Convert this part to an integer
            
            faces.append(imageNP)
            ids.append(id)
        except Exception as e:
            print(f"Error parsing the filename: {image_path}. Skipping this image. Error: {e}")

    if len(faces) > 0 and len(ids) > 0:
        ids = np.array(ids)
        clf = cv.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")
        print("Classifier trained and saved.")
    else:
        print("No faces found. Training failed.")


train_classifier("C:/Users/HP/Desktop/New folder/rohit/data")

# Load the classifier for face recognition
clf = cv.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# Load the pre-trained model for uniform detection
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('E:/computer_vision/basedata/traning',
                                          target_size=(200, 200),
                                          batch_size=2,
                                          class_mode='binary')

validation_dataset = train.flow_from_directory('E:/computer_vision/basedata/validation',
                                               target_size=(200, 200),
                                               batch_size=2,
                                               class_mode='binary')

# Model architecture for uniform detection
model = tf.keras.models.Sequential([ 
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])
    # Your CNN architecture here (unchanged)
model.fit(train_dataset,
                    steps_per_epoch=30,
                    epochs=30,
                    validation_data=validation_dataset)



# Function to recognize faces in the image
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    coords = []  # To store the coordinates of the detected faces
    face_label = "Unknown"  # Default label in case no recognized face is found

    for (x, y, w, h) in features:
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)  # Draw rectangle around face

        # Extract the face region
        face_region = gray_image[y:y + h, x:x + w]
        id, pred = clf.predict(face_region)

        # Confidence calculation based on the prediction distance
        confidence = int(100 * (1 - pred / 300))  # The confidence increases as pred decreases
        print(f"Predicted ID: {id}, Confidence: {confidence}")
        # Check if confidence is above a certain threshold (set to 71% here)
        if confidence > 72:
            if id == 1:  # Assuming '1' corresponds to 'Rohit'
               face_label = "Rohit"
            elif id == 2:  # Assuming '2' corresponds to 'Neeraj'
                face_label = "Neeraj"
            elif id == 3:  # Assuming '3' corresponds to 'Person 3'
                face_label = "Person 3"
            elif id == 4:  # Assuming '4' corresponds to 'Person 4'
                face_label = "Person 4"
            else:
                face_label = "Unknown"

        coords = [x, y, w, h]

    return coords, face_label  # Return both coordinates and the face label


def recognize(img, clf, facecascade):
    coords, face_label = draw_boundary(img, facecascade, 1.1, 10, (255, 255, 255), "face", clf)
    return img, face_label  # Return the image and the face label


# Load the pre-trained face cascade classifier
facecascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# Capture video from the webcam
url = "http://192.168.124.182:4747/video"   # Example IP, replace with your IP and use 'http' if 'https' doesn't work

# Open the video stream from the IP camera
cap = cv.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Couldn't connect to the camera.")
    exit()

# Open CSV file to save the results
csv_file_path = 'detection_res.csv'

# Dictionary to track attendance for each person (avoiding multiple marks in a single day)
marked_today = {}

# Helper function to check if attendance has already been marked today for a specific person
def is_attendance_marked_today(person_name):
    today = datetime.today().strftime('%Y-%m-%d')  # Get today's date in YYYY-MM-DD format
    if person_name not in marked_today:
        marked_today[person_name] = today
        return False  # Attendance is not marked today
    elif marked_today[person_name] == today:
        return True  # Attendance is already marked for today
    return False

with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)

    # If the file is empty, write the headers
    if file.tell() == 0:
        writer.writerow(['Timestamp', 'Label', 'Uniform Status'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame for face recognition
        img, face_label = recognize(frame, clf, facecascade)

        # Resize the frame to match the input size of the model for uniform detection
        face_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        face_rgb_resized = cv.resize(face_rgb, (200, 200))  # Resize image to match the input size for uniform model
        img_array = image.img_to_array(face_rgb_resized)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Rescale image

        # Predict uniform status using the trained uniform detection model
        uniform_prediction = model.predict(img_array) 
        uniform_status = "In Uniform" if uniform_prediction < 0.5 else "Not in uniform"

        # Concatenate both face label and uniform status into one string
        result_text = f"{face_label} - {uniform_status}"

        # Display the concatenated result on the frame
        cv.putText(frame, result_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Check if attendance has already been marked today for this person
        if not is_attendance_marked_today(face_label):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get current timestamp
            writer.writerow([timestamp, face_label, uniform_status])
            print(f"Attendance marked for {face_label}. Uniform status: {uniform_status}")
        else:
            print(f"Your attendance is already marked today, {face_label}.")

        # Resize the frame to 200x200 before displaying
        frame_resized = cv.resize(frame, (600,600))

        # Display the resulting frame with face recognition and uniform status
        cv.imshow("Live Camera Feed", frame_resized)

        # Exit if 'q' is pressed
        if cv.waitKey(28) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
