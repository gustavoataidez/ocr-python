import cv2

for i in range(5):  # Tente os primeiros 5 índices de câmera
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Câmera encontrada no índice {i}")
        cap.release()
        break
else:
    print("Nenhuma câmera disponível nos índices 0-4.")
