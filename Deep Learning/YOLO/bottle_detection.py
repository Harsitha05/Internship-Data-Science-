from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Loop through detections
    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])

            # Get object name
            class_name = model.names[cls]

            # Detect ONLY bottle
            if class_name == "bottle":

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # Label
                cv2.putText(frame,
                            "Bottle",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0),
                            2)

    # Show webcam
    cv2.imshow("Bottle Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()