import cv2
import numpy as np
import pytesseract

# Configuração do Tesseract (adicione o caminho do executável, se necessário)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Carregar o modelo YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Ajuste para obter as camadas de saída de forma compatível com diferentes versões do OpenCV
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Definir classes (ou carregue de um arquivo, se tiver)
classes = ["license_plate"]

# Função para realizar OCR em uma região da placa detectada
def reconhecer_texto(placa_roi):
    # Verifica se a região da placa não está vazia
    if placa_roi is not None and placa_roi.size > 0:
        # Converter para tons de cinza e aplicar binarização para melhorar o OCR
        placa_roi = cv2.cvtColor(placa_roi, cv2.COLOR_BGR2GRAY)
        placa_roi = cv2.threshold(placa_roi, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        config = r'--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        texto = pytesseract.image_to_string(placa_roi, config=config)
        return texto.strip()
    return ""

# Inicializar a captura de vídeo
video_capture = cv2.VideoCapture(0)

while True:
    # Ler um quadro da câmera
    ret, frame = video_capture.read()
    if not ret:
        print("Erro ao capturar o quadro.")
        break

    # Preparar a imagem para o YOLO
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    placas_detectadas = []

    # Processar detecções
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Apenas a classe de placa e confiança alta
            if confidence > 0.5:  # Ajuste conforme necessário
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Garantir que as coordenadas estão dentro dos limites da imagem
                if x >= 0 and y >= 0 and x + w <= width and y + h <= height:
                    # Extrair a região da placa e aplicar OCR
                    placa_roi = frame[y:y + h, x:x + w]
                    texto = reconhecer_texto(placa_roi)
                    placas_detectadas.append((texto, (x, y, w, h)))

    # Exibir os resultados
    for texto, (x, y, w, h) in placas_detectadas:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, texto, (x + 5, y - 5), font, 0.5, (255, 255, 255), 1)

    # Exibir o quadro com as detecções
    cv2.imshow('Detecta de Placa com OCR', frame)

    # Verificar se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar a janela
video_capture.release()
cv2.destroyAllWindows()
