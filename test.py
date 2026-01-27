import base64
import glob
import os
import requests
import logging

logging.getLogger().setLevel(logging.DEBUG)

# 1. 설정
url = "http://localhost:8000/detect"
message = "Batch Process"
input_dir = "inputs"
output_dir = "outputs"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

search_path = os.path.join(input_dir, "*.jpg")
image_files = glob.glob(search_path)

print(f"총 {len(image_files)}개의 파일을 찾았습니다.")

# 3. 파일 루프 실행
for file_path in image_files:
    filename = os.path.basename(file_path)

    print(f"처리 중: {filename}...", end=" ", flush=True)

    try:
        # 파일 읽기 및 전송
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"message": message}
            response = requests.post(url, data=data, files=files)

        # 응답 처리
        if response.status_code == 200:
            result = response.json()
            # base64 디코딩 및 저장
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "wb") as f_out:
                f_out.write(base64.b64decode(result["image"]))
            print("Done!")
        else:
            print(f"에러 (상태코드: {response.status_code})")

    except Exception as e:
        print(f"실패: {e}")

print("\n모든 작업이 완료되었습니다.")
