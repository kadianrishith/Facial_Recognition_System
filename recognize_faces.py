import os
import csv
from datetime import datetime

import cv2

# Load Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# Load Haar Cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Camera
cam = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX

# ID -> Name
names = {}
for index, person in enumerate(sorted(os.listdir("dataset"))):
    person_path = os.path.join("dataset", person)
    if os.path.isdir(person_path):
        names[index] = person.replace("_", " ").title()

marked_people = set()

# Record attendance only once per person for this run

print("Face Recognition Started...")
print("Press 'q' to Exit")

while True:

    ret, img = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0),2)

        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 90:

            name = names.get(id, "Unknown")

            cv2.putText(
                img,
                name,
                (x, y-10),
                font,
                0.8,
                (0,255,0),
                2
            )

            if name != "Unknown" and name not in marked_people:
                marked_people.add(name)
                now = datetime.now()

                with open("attendance.csv","a",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, now.strftime("%Y-%m-%d %H:%M:%S")])

        else:

            cv2.putText(
                img,
                "Unknown",
                (x,y-10),
                font,
                0.8,
                (0,0,255),
                2
            )

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()