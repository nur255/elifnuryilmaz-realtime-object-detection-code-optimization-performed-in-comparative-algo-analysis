#SSD300 video detection

import torch
import torchvision
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

# Model ve sınıf isimlerini yükle
weights = torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
model = torchvision.models.detection.ssd300_vgg16(weights=weights).eval()

# Cihazı ayarla (CUDA varsa GPU, yoksa CPU kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class_names = weights.meta["categories"]

# Video dosyasını aç
video_path = 'D:/MSc/4.mp4'  # Video dosyanızın yolu
cap = cv2.VideoCapture(video_path)

# Video kaydetme için kod
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (640, 480))

fps_list = []
frame_count = 0
start_time = time.time()
confidence_threshold = 0.5  # Eşik değeri

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü tensor yapısına dönüştür
    image_tensor = F.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    # Modeli kullanarak tahmin yap
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # Tespit edilen nesneleri çiz
    for i, box in enumerate(predictions['boxes']):
        score = predictions['scores'][i].item()
        if score > confidence_threshold:  # Eşik değeri üzerinde olanları seç
            x1, y1, x2, y2 = box.int().tolist()
            label = class_names[int(predictions['labels'][i])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS hesapla ve ekrana yazdır
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    fps_list.append(fps)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüyü video dosyasına kaydet
    out.write(frame)

    # Görüntüyü ekranda göster
    cv2.imshow("SSD Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Performans metriklerini hesapla
mAP = np.mean([s.item() for s in predictions['scores'] if s > confidence_threshold])

# Grafik çiz
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(fps_list, label="FPS")
plt.xlabel("Frame")
plt.ylabel("FPS")
plt.legend()
plt.title("Real-time FPS Over Time")

plt.subplot(1, 2, 2)
plt.bar(["mAP"], [mAP], color='blue')
plt.ylabel("mAP")
plt.title("Average Precision (mAP)")

plt.tight_layout()
plt.show()

# Kaynakları serbest bırak
cap.release()
out.release()
cv2.destroyAllWindows()
