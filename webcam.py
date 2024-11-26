import cv2
import pytesseract

# Defina o caminho do executável do Tesseract, caso necessário
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detectar_placas(frame):
    # Reduz o tamanho do quadro para melhorar desempenho
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    # Encontrar contornos
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    placas_detectadas = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Considera apenas contornos quadrilaterais
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)

            # Filtrar por proporção e área
            if 500 < area < 5000 and 2 < aspect_ratio < 5:
                placa_roi = gray[y:y + h, x:x + w]
                
                # Melhorar o contraste da imagem da placa
                placa_roi = cv2.threshold(placa_roi, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # OCR na região da placa, configurado para um padrão específico
                config = r'--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                texto = pytesseract.image_to_string(placa_roi, config=config)

                # Filtrar por padrão da placa brasileira
                if len(texto) >= 7:
                    placas_detectadas.append((texto.strip(), (x * 2, y * 2, w * 2, h * 2)))  # Coordenadas redimensionadas para o tamanho original

    return placas_detectadas

# Captura de vídeo
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Erro ao acessar a webcam. Verifique se ela está conectada e tente novamente.")
    exit()

processar_este_quadro = True

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Falha ao capturar a imagem da webcam.")
        break

    if processar_este_quadro:
        placas_detectadas = detectar_placas(frame)
    processar_este_quadro = not processar_este_quadro

    # Exibir os resultados
    for texto, (x, y, w, h) in placas_detectadas:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, texto, (x + 5, y - 5), font, 0.5, (255, 255, 255), 1)
        print(f"Placa detectada: {texto}")
        print("Encerrando o programa...")
        video_capture.release()
        cv2.destroyAllWindows()
        exit()

    # Exibir a imagem
    cv2.imshow('Webcam_license_plate_recognition', frame)

    # Pressionar 'q' para sair manualmente
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
