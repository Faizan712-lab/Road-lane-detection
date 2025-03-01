cap = cv2.VideoCapture('video.mp4')  # Replace with your video file path
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame = process_frame(frame)
    cv2.imshow('Enhanced Road Lane Detection', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
