import cv2

backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, 0)
indices = (0, 1, 2)

for backend in backends:
    for index in indices:
        cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            print('backend', backend, 'index', index, 'opened', cap.isOpened(), 'ret', ret, 'shape', None if frame is None else frame.shape)
            cap.release()
            raise SystemExit(0)

print('no camera available')
