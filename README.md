# Facial Recognition System

A Python-based facial recognition attendance system that captures faces, trains a recognizer, and marks attendance automatically using a webcam.

## Intern ID
CITS2502

## Description
This project uses OpenCV and Python to:
- Capture face images from a webcam
- Store images in a dataset folder
- Train a face recognition model
- Recognize faces in real time
- Record attendance in a CSV file

## Features
- Real-time face detection using Haar Cascade
- Face image collection for training
- Model training with LBPH recognizer
- Live face recognition and attendance marking
- Attendance logging to CSV
- Simple and beginner-friendly project structure

## Technologies Used
- Python
- OpenCV
- NumPy
- Pillow
- CSV

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project folder:
   ```bash
   cd Facial-Recognition-System
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the face capture script to collect images:
   ```bash
   python capture_faces.py
   ```
2. Train the model:
   ```bash
   python train_model.py
   ```
3. Start recognition and attendance logging:
   ```bash
   python recognize_faces.py
   ```

## Folder Structure
```text
Facial-Recognition-System/
├── capture_faces.py
├── train_model.py
├── recognize_faces.py
├── dataset/
├── trainer/
├── attendance.csv
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
```

## Screenshots
- Webcam-based face capture
- Real-time face recognition window
- Attendance entries saved in attendance.csv

## Notes
- Make sure your webcam is connected and allowed by your system.
- The recognition accuracy improves with more training images.
