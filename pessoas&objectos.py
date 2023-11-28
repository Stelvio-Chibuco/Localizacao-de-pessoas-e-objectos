import cv2
import numpy as np

# Caminhos para os arquivos de configuração, pesos e nomes das classes
config_path = "/home/stelvio/darknet/cfg/yolov4.cfg"
weights_path = "/home/stelvio/darknet/cfg/yolov4.weights"
class_names_path = "/home/stelvio/darknet/cfg/coco.names"

# Carregar os nomes das classes
with open(class_names_path, "r") as f:
    class_names = f.read().strip().split("\n")

# Carregar o modelo YOLO
net = cv2.dnn.readNet(weights_path, config_path)

# Inicializar a captura de vídeo
video_path = "/home/stelvio/Videos/video.mp4"
cap = cv2.VideoCapture(video_path)
# Verificar se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

while True:
    ret, frame = cap.read()

    # Verificar se o frame foi lido corretamente
    if not ret:
        break

    # Converta o frame para escala de cinza (dlib trabalha em imagens em escala de cinza)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Obter as dimensões do frame
    height, width = frame.shape[:2]

    # Preparar o frame para a detecção
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Obter as detecções
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Processar as detecções
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            if confidence > 0.5:  # Limiar de confiança
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordenadas do canto superior esquerdo do retângulo
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Desenhar o retângulo e a etiqueta da classe
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{class_names[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar o frame com as detecções
    cv2.imshow("Detector de Veiculos", frame)

    # Verificar se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
