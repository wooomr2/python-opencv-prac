from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import io
import base64
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

app = FastAPI()

model = YOLO("yolov8n.pt")  # YOLOv8 모델 로드


# 데이터 모델 정의
class DetectionResult(BaseModel):
    message: str
    image: str


def detect_objects(image: Image):
    img = np.array(image)  # 이미지를 numpy 배열로 변환
    results = model(img)  # 객체 탐지
    class_names = model.names  # 클래스이름 저장

    # 결과를 바운딩 박스, 클래스이름, 정확도로 이미지에 표시
    result = results[0]  # 탐색한 이미지가 하나임
    boxes = result.boxes.xyxy  # 바운딩 박스
    confidences = result.boxes.conf  # 신뢰도
    class_ids = result.boxes.cls  # 클래스 아이디
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)  # 좌표를 정수로 변환
        label = class_names[int(class_id)]  # 클래스 이름

        print(x1, y1, x2, y2, label)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img,
            f"{label} {confidence:.2f}",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

    result_image = Image.fromarray(img)  # 결과 이미지를 PIL로 변환
    return result_image


@app.get("/")
async def index():
    return {"message": "Hello FastAPI"}


@app.post("/detect", response_model=DetectionResult)
async def detect_service(message: str = Form(...), file: UploadFile = File(...)):
    # 이미지를 읽어서 PIL 이미지로 변환
    image = Image.open(io.BytesIO(await file.read()))

    if image.mode == "RGBA":  # 알파 채널 제거하고 RGB로 변환
        image = image.convert("RGB")
    elif image.mode != "RGB":  # RGB 모드가 아닌 경우도 RGB로 변환
        image = image.convert("RGB")

    result_image = detect_objects(image)  # 객체 탐지 수행

    # 이미지 결과를 base64 인코딩
    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return DetectionResult(message=message, image=img_str)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
