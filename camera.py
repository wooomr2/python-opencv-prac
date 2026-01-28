import base64
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import json
from ultralytics import YOLO

# model = YOLO("yolov8n.pt")
model = YOLO("numbers-best.pt")

broker = "localhost"
port = 1883
topic = "/camera/objects"


client = mqtt.Client()


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")


client.on_connect = on_connect
client.connect(broker, port)


def get_colors(num_colors):
    np.random.seed(0)
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_colors)]

    return colors


class_names = model.names
num_classes = len(class_names)
colors = get_colors(num_classes)


def detect_objects(image: np.array):
    results = model(image, verbose=False)  # 객체 탐지
    class_names = model.names  # 클래스 이름 저장

    # 결과를 바운딩 박스와 정확도로 이미지에 표시
    for result in results:
        boxes = result.boxes.xyxy  # 바운딩 박스
        confidences = result.boxes.conf  # 신뢰도
        class_ids = result.boxes.cls  # 클래스
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)  # 좌표를 정수로 변환
            label = class_names[int(class_id)]  # 클래스 이름
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[int(class_id)], 2)
            cv2.putText(
                image,
                f"{label} {confidence:.2f}",
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                colors[int(class_id)],
                2,
            )

    return image


# cam_addr = "rtsp://admin:admin123@203.24.58.146:8888/cam1"
cam_addr = 0
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result_image = detect_objects(frame)

    # 이미지 결과를 base64 인코딩
    _, buffer = cv2.imencode(".jpg", result_image)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")

    # 객체 탐지 이미지 전송
    payload = json.dumps({"image": jpg_as_text})
    client.publish(topic, payload)

    # 프레임을 화면에 표시
    cv2.imshow("Frame", result_image)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# 리소스 해제
cap.release()
cv2.destroyAllWindows()
client.disconnect()
