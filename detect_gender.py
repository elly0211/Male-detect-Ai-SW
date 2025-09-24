from ultralytics import YOLO
import cv2, requests

print("🟢 성별 분류 모델 로딩 중...")
model = YOLO("best.pt")
print("✅ 모델 로드 완료!")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라 접근 실패!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    classes = results[0].boxes.cls.cpu().numpy()

    for cls in classes:
        if int(cls) == 0:
            print("🚨 남성 감지! 서버로 전송...")
            try:
                requests.post("http://localhost:8080/WomanOnlyTrainGuard/alert", data={"gender":"male"}, timeout=2)
                print("✅ 전송 성공")
            except Exception as e:
                print("❌ 전송 실패:", e)

    cv2.imshow("👁 Gender Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()