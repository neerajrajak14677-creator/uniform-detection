import cv2 as cv

def generate_dataset():
    face_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def face_cropped(img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:  # If no faces are detected
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face
    
    cap = cv.VideoCapture(0)
    id = 4
    img_id = 0

    while True:

    
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no frame is returned
        
        face = face_cropped(frame)
        if face is not None:
            img_id += 1
            face = cv.resize(face, (500, 500))  # Resize the face image
            face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)  # Convert to grayscale
            file_name_path = f"C:/Users/HP/Desktop/New folder/rohit/data/user{id}.{img_id}.jpg"  # Use formatted string
            cv.imwrite(file_name_path, face)  # Save the image
            cv.putText(face, str(img_id), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Put img_id on the image
            cv.imshow("cropped_image", face)  # Show the image
            
            if cv.waitKey(1) == 13 or img_id == 200:  # 13 is the Enter key ASCII code
                break  # Exit the loop after 100 images or on pressing Enter
    
    cap.release()
    cv.destroyAllWindows()
    print("Collecting samples is completed.")

# Call the function to generate the dataset
generate_dataset()
