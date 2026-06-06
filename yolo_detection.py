#yolo video detection

import torch
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Modeli yükleme
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolo11s.pt').to(device)

# Video dosyasının yolu
video_path = 'D:/MSc/4.mp4'  # Buraya video dosyanızın yolunu girin

# Video açma
cap = cv2.VideoCapture(video_path)

# Eğer video dosyası açılmadıysa hata mesajı ver
if not cap.isOpened():
    print("Video dosyası açılamadı.")
    exit()

# Çıktı görüntülerini kaydedeceğimiz klasör
output_dir = 'output_frames'
os.makedirs(output_dir, exist_ok=True)

plt.ion()

# Performans metriklerini saklamak için listeler
fps_list = []
confidence_list = []

frame_count = 0  # Frame sayısını tutmak için değişken

# Gerçek zamanlı görüntüleme
while True:
    ret, frame = cap.read()
    if not ret:
        print("Videonun sonuna gelindi.")
        break
    
    frame_count += 1
    
    # Zaman ölçümünü başlat
    start_time = time.time()
    
    # Nesne tespiti
    results = model(frame)
    
    # FPS hesaplama
    fps = 1 / (time.time() - start_time)
    fps_list.append(fps)

    # Güven oranını hesapla
    if results[0].boxes is not None:  # Nesne bulunmadığında None olabilir, kontrol edelim
        confidences = results[0].boxes.conf.cpu().numpy()  # Güven değerleri numpy array olarak
        mean_confidence = np.mean(confidences) if confidences.size > 0 else 0  # Boş olma durumu için kontrol
    else:
        mean_confidence = 0  # Nesne yoksa 0
    confidence_list.append(mean_confidence)
    
    # Sonuçları görselleştirme
    annotated_frame = results[0].plot()  # Görüntü üzerine tespit edilen nesneleri çiz
    
    # Her 10 frame'de bir kaydet
    if frame_count % 10 == 0:
        output_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        cv2.imwrite(output_path, annotated_frame)
    
    # FPS ve güven oranını görüntüye ekle
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Confidence: {mean_confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Görüntüyü göster
    cv2.imshow('YOLOv11s Video Object Detection', annotated_frame)
    
    
    # Kaydedilecek frame dosyasının yolu
    output_path = os.path.join(output_dir, f'v11l_frame_{frame_count}.jpg')

    # Görüntüyü dosyaya kaydet
    cv2.imwrite(output_path, annotated_frame)
    

    # Çıkış kontrolü
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Performans metriklerini görselleştirme
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# FPS grafiği
ax1.plot(fps_list, label='FPS')
ax1.set_title('Frames Per Second (FPS)')
ax1.set_xlabel('Frame')
ax1.set_ylabel('FPS')
ax1.legend()

# Güven oranı grafiği
ax2.plot(confidence_list, label='Mean Confidence', color='orange')
ax2.set_title('Mean Detection Confidence')
ax2.set_xlabel('Frame')
ax2.set_ylabel('Confidence')
ax2.legend()

plt.tight_layout()
plt.savefig('performance_metrics.png')
plt.show(block=False)


# Video dosyasını ve OpenCV penceresini kapatma
cap.release()
cv2.destroyAllWindows()
