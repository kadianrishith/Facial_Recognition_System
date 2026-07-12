import os
import sys

import cv2


def sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip()
    return cleaned or "person"


def main() -> None:
    name = sys.argv[1].strip() if len(sys.argv) > 1 else input("Enter your name: ").strip()
    name = sanitize_name(name)

    path = os.path.join("dataset", name)
    os.makedirs(path, exist_ok=True)

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if detector.empty():
        print("Error: Could not load face detection model.")
        sys.exit(1)

    camera = None
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, 0):
        for index in (0, 1, 2):
            candidate = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if candidate.isOpened():
                ret, _ = candidate.read()
                if ret:
                    camera = candidate
                    print(f"Camera connected using backend {backend} and index {index}.")
                    break
            candidate.release()
        if camera is not None:
            break

    if camera is None or not camera.isOpened():
        print("Cannot access camera. Please check your webcam connection and camera permissions.")
        sys.exit(1)

    count = 0

    print("Looking at camera...")
    print("Collecting face images...")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Cannot read from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                face = gray[y:y+h, x:x+w]
                filename = os.path.join(path, f"{count}.jpg")
                cv2.imwrite(filename, face)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Images: {count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Face Capture", frame)

            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break

            if count >= 100:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\n100 images saved successfully in {path}")


if __name__ == "__main__":
    main()