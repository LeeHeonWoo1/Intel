import cv2

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1290)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

while True:
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
        cv2.imshow("VideoFrame", binary)
        if cv2.waitKey(33) == ord('q'): break
    else: break

