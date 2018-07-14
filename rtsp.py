import cv2
#rtsp://admin:admin12345@192.168.1.247:554/mpeg4
cap = cv2.VideoCapture("rtsp://admin:admin12345@192.168.1.249:554/mpeg4")
print(cap.isOpened())
ret,frame = cap.read()
cv2.imshow("frame", frame)