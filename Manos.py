# LIBRERIAS
import cv2
import mediapipe as mp

# --- 1) CONFIGURACION DE MEDIA PIPE ---
mp_drawing = mp.solutions.drawing_utils # Utilidades para dibujar los puntos y conexiones
mp_hands = mp.solutions.hands           # Solución de MediaPipe para detección y seguimiento de manos

# Hands() permite ajustar parametros como el numero maximo de manos detectadas y la confianza
hands = mp_hands.Hands(
    static_image_mode=False,      # Si es False, se detecta y hace seguimiento en video. Si es True, se detecta cada frame sin seguimiento.
    max_num_hands=2,              # maximo de manos a detectar
    min_detection_confidence=0.5, # confianza minima para considerar una deteccion valida
    min_tracking_confidence=0.5   # confianza minima para considerar un seguimiento valido (solo si static_image_mode=False)
)

# --- 2) ABRIR CAMARA (0 por lo general, cambia si tienes multiples webcams) ---
cap = cv2.VideoCapture(0)  # Cambia el índice si tienes varias cámaras o si la cámara principal no es la 0

if not cap.isOpened(): # Verificar que la cámara se abrió correctamente
    raise RuntimeError("No se puede abrir la cámara. Revisa el índice (0, 1, ...) y que no esté en uso.")

# --- 3) BUCLE PRINCIPAL ---
# Colores para dibujar los puntos y conexiones
landmark_color = (0, 0, 255)    # Color de los puntos (landmarks)
connection_color = (0, 0, 255)  # Color de las conexiones entre puntos

while True:                 # Leer un frame de la cámara while True es un bucle infinito que se ejecuta hasta que se presiona ESC para salir
    ret, frame = cap.read() # frame es la imagen capturada, ret es un booleano que indica si la captura fue exitosa.
    if not ret:             # Si no se pudo leer el frame, salir del bucle (por ejemplo, si la cámara se desconecta)
        break               # break sale del bucle while, lo que llevará a liberar recursos y cerrar la aplicación

    # Mediapipe funciona en RGB y OpenCV usa BGR, asi que convertimos

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # esto convierte la imagen de BGR a RGB para que MediaPipe pueda procesarla correctamente
   

    # Procesar la imagen con MediaPipe para detectar manos y obtener los landmarks
    results = hands.process(frame_rgb) # sirve para detectar las manos en la imagen y obtener los puntos de referencia (landmarks) de cada mano detectada. El resultado se almacena en la variable results, que contiene información sobre las manos detectadas y sus landmarks.

    # Si se detectan manos, dibujar los landmark y conexiones
    if results.multi_hand_landmarks: 
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar puntos y conexiones sobre la imagen original
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=connection_color, thickness=2)
            )

            # Ejemplo: tomar la punta del dedo indice (landmark 8) y mostrar sus coordenadas
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            cv2.putText(frame, f"Indice: ({x},{y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Mostrar la imagen con las manos proyectadas
    cv2.imshow("Camara - Manos con MediaPipe", frame)

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- 4) LIBERAR RECURSOS ---
cap.release()
cv2.destroyAllWindows()

# Cerrar el detector de manos
hands.close()



