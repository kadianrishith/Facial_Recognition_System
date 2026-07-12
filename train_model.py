import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = "dataset"

faces = []
labels = []
label_map = {}

current_id = 0

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    label_map[current_id] = person

    for image in os.listdir(person_path):

        image_path = os.path.join(person_path, image)

        gray = Image.open(image_path).convert("L")

        image_np = np.array(gray, "uint8")

        faces.append(image_np)
        labels.append(current_id)

    current_id += 1

recognizer.train(faces, np.array(labels))

if not os.path.exists("trainer"):
    os.makedirs("trainer")

recognizer.save("trainer/trainer.yml")

print("Training completed successfully!")