import cv2
import pickle
import pandas as pd
from datetime import datetime
import os

# Load Model
with open("data/model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Face Detector
facedetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start Camera
video = cv2.VideoCapture(0)

attendance = []

while True:
    ret, frame = video.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        resized_img = resized_img.reshape(1, -1)

        name = model.predict(resized_img)[0]

        current = datetime.now()
        date = current.strftime("%d-%m-%Y")
        time = current.strftime("%H:%M:%S")

        # Screen Text
        display_text = f"{name} | {date} | {time}"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            display_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        # Save only once per person
        if name not in [row[0] for row in attendance]:
            attendance.append([
                name,
                date,
                time
            ])

    cv2.imshow("Face Recognition Attendance System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Create Attendance Folder
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# Save CSV
df = pd.DataFrame(
    attendance,
    columns=["Name", "Date", "Time"]
)

filename = (
    "Attendance/Attendance_"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    + ".csv"
)

df.to_csv(filename, index=False)

print("Attendance Saved Successfully!")
print("File:", filename)