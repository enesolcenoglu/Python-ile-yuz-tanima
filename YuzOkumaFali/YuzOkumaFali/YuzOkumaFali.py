import cv2
import mediapipe as mp
import numpy as np
import random

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

fal = ["Bugun sans seninle olacak ","Kendine guven ve pes etme ","Bereketli bir gun"]

rastgele = random.choice(fal)

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # Kamera yakalama döngüsü
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            break

        # Görüntüyü RGB formatına dönüştür
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe Yüz Tespiti ile görüntüyü işle
        results = face_detection.process(image)

        # Tespit sonuçlarını görüntü üzerine çiz
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

                # Eğer yüz tespit edildiyse, beyaz bir pencere oluştur ve falı
                # yazdır
                white_image = 255 * np.ones((200, 325, 3), dtype=np.uint8)
                cv2.putText(white_image, rastgele, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.imshow('Beyaz Pencere', white_image)

        # Görüntüyü BGR formatına geri dönüştür
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Görüntüyü göster
        cv2.imshow('Yüz Tespiti', image)

        # ESC tuşuna basarak çıkış yap
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

