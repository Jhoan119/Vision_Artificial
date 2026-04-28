import cv2
import mediapipe as mp
import os

# ----------------------------- Configuración ---------------------------------
nombre = 'Letra_U'  # Cambia a Letra_E, Letra_I, Letra_O, Letra_U según lo que captures
direccion = 'C:/Users/Admin/Desktop/Aprendiendo python 1/Entrenamiento'
carpeta = direccion + '/' + nombre

# Crear carpeta si no existe
if not os.path.exists(carpeta):
    print('Carpeta creada:', carpeta)
    os.makedirs(carpeta)

cont = 0  # Contador de imágenes guardadas

# ----------------------------- Cámara ----------------------------------------
cap = cv2.VideoCapture(0)

# ----------------------------- MediaPipe -------------------------------------
mp_hands = mp.solutions.hands
manos = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
dibujo = mp.solutions.drawing_utils

# ----------------------------- Bucle principal -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])

            dibujo.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)

        # Si hay puntos de mano detectados
        if len(posiciones) != 0:
            # Punto central (ID 9)
            cx, cy = posiciones[9][1], posiciones[9][2]

            # Tamaño del recorte
            t = 120

            x1, y1 = cx - t, cy - t
            x2, y2 = cx + t, cy + t

            # Evitar coordenadas negativas
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            dedos_reg = copia[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Guardar imagen si el recorte es válido
            if dedos_reg.size != 0:
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f"{carpeta}/Mano_{cont}.jpg", dedos_reg)
                print("Imagen guardada:", cont)
                cont += 1

    cv2.imshow("Captura", frame)

    # Tecla ESC para salir o cuando llegue a 300 imágenes
    k = cv2.waitKey(1)
    if k == 27 or cont >= 300:
        break

cap.release()
cv2.destroyAllWindows()