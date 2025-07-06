import cv2
from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    flipped = cv2.flip(frame, 1)

    results = model.predict(flipped, verbose=False)
    results[0].names = {
        0:'knife'
    }
    boxes = results[0].boxes
    names = results[0].names  

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{names[cls_id]} {conf:.2f}"

        if conf >= 0.65:
            cv2.rectangle(flipped, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.putText(flipped, label, (x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 0), thickness=1)

    cv2.imshow("Webcam", flipped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
