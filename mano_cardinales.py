import math # es una librería necesaria para calcular el ángulo de dirección de la mano 
import cv2 # OpenCV para capturar video y dibujar sobre la imagen
import mediapipe as mp # MediaPipe para detección y seguimiento de manos

""" vector es una representación matemática que tiene una magnitud y una dirección. En este caso, 
    se utiliza para representar la dirección de la mano basada en la posición de la muñeca y la punta del dedo medio. """

""" LANDMARKS son puntos de referencia específicos en la mano que MediaPipe detecta. 
    Cada landmark tiene una posición (x, y, z) relativa a la imagen."""    


# Iterar es una técnica de programación que permite repetir un bloque de código varias veces. 

def compute_hand_direction(landmarks, image_shape): # define una función que toma los landmarks de la mano y el tamaño de la imagen para calcular la dirección de la mano
    """Calcula dirección de la mano usando la muñeca y la punta del dedo medio.

    Devuelve:
      - angle_deg: ángulo en grados (0 = Este, 90 = Norte, 180 = Oeste, 270 = Sur)
      - wrist_pt, tip_pt: coordenadas en píxeles para dibujar.
    """
    h, w, _ = image_shape # altura, ancho, canales de la imagen
    wrist = landmarks[0] # muñeca wrist es el primer landmark (índice 0) que representa la posición de la muñeca
    tip = landmarks[12]  # punta del dedo medio tip es el landmark número 12 que representa la punta del dedo medio

    wx, wy = int(wrist.x * w), int(wrist.y * h) 
    tx, ty = int(tip.x * w), int(tip.y * h)

    dx = tx - wx # calcula la direccion del vector desde la muñeca hasta la punta del dedo medio, dx es la diferencia en x y
    dy = ty - wy # diferencia en y. Este vector representa la dirección de la mano.

    # En OpenCV el eje y apunta hacia abajo, invertimos dy para convertir al sistema matemático.
    angle_rad = math.atan2(-dy, dx) # es basicamente
    angle_deg = (math.degrees(angle_rad) + 360) % 360

    return angle_deg, (wx, wy), (tx, ty)


def open_camera(index: int): # función para abrir la cámara con el índice especificado y verificar que se abrió correctamente
    cap = cv2.VideoCapture(index) # intenta abrir la cámara con el índice dado (0, 1, etc.). Si tienes varias cámaras, puedes cambiar el índice para seleccionar la cámara deseada.
    if not cap.isOpened(): # verifica si la cámara se abrió correctamente. Si no se pudo abrir, imprime un mensaje de error y devuelve None para indicar que no se pudo acceder a la cámara.
        print(f"No se pudo abrir la cámara {index}")
        return None
    return cap

# la funcion hand_label_es convierte las etiquetas de mano "Left" y "Right" a "Izq" y "Der" respectivamente, para mostrar en español.

def _hand_label_es(label: str) -> str:
    """Convierte 'Left'/'Right' a etiquetas cortas en español."""
    label = label.lower() # convierte la etiqueta a minúsculas para facilitar la comparación, ya que MediaPipe devuelve las etiquetas en mayúscula inicial (e.g., "Left", "Right").
    if label == "left": #
        return "Izq"
    if label == "right":
        return "Der"
    return label.capitalize() #label.capitalize() devuelve la cadena con la primera letra en mayúscula y el resto en minúscula, lo que es útil para mantener un formato consistente en caso de que se reciba una etiqueta diferente o inesperada.


# La función _hand_bounding_box calcula un rectángulo que contiene todos los landmarks de la mano, con un padding opcional para agregar espacio alrededor de la mano. Esto es útil para dibujar una caja alrededor de la mano detectada en la imagen.

def _hand_bounding_box(landmarks, image_shape, padding: int = 20):
    """Calcula el rectángulo que contiene todos los landmarks de la mano."""
    h, w, _ = image_shape
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w, max(xs) + padding)
    y2 = min(h, max(ys) + padding)
    return x1, y1, x2, y2

""" def main() es la función principal que se ejecuta al iniciar el programa. 
    Dentro de esta función, se abre la cámara, se inicializa el detector de manos de MediaPipe, 
    y se entra en un bucle donde se captura cada frame de video, se procesa para detectar las manos y sus landmarks, 
    y se dibujan los resultados sobre la imagen. El programa continúa ejecutándose hasta que el usuario presiona la tecla ESC para salir. """

def main():
    cam_index = 1
    cap = open_camera(cam_index)
    if cap is None:
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Detectar hasta 2 manos
        model_complexity=1, # para usar un modelo más preciso (0 es el más rápido, 1 es el más preciso)
        min_detection_confidence=0.7, # confianza mínima para considerar una detección válida. Si la confianza de la detección es menor que este valor, la mano no se considerará detectada.
        min_tracking_confidence=0.7,  # confianza mínima para considerar un seguimiento válido. Si la confianza de seguimiento es menor que este valor, la mano no se considerará rastreada.
    )
    mp_draw = mp.solutions.drawing_utils # Utilidades para dibujar los puntos y conexiones de la mano detectada en la imagen.

    while True: # Bucle principal para capturar video y procesar cada frame
        ret, frame = cap.read()
        if not ret:
            break

# Convertir la imagen de BGR (formato de OpenCV) a RGB (formato requerido por MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:  # Si se detectan manos, iterar sobre cada mano detectada y su información de orientación (izquierda/derecha)
            for hand_landmarks, hand_handedness in zip( # zip se utiliza para repetir simultáneamente sobre dos listas:
                result.multi_hand_landmarks, result.multi_handedness
            ):
                # Dibujar puntos y conexiones
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Caja alrededor de la mano
                x1, y1, x2, y2 = _hand_bounding_box(hand_landmarks.landmark, frame.shape)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Etiqueta (Izq/Der + confianza)
                label = _hand_label_es(hand_handedness.classification[0].label)
                score = hand_handedness.classification[0].score
                text = f"{label} {score:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # (Opcional) Contar manos detectadas:
        # if result.multi_hand_landmarks:
        #     cv2.putText(frame, f"Manos: {len(result.multi_hand_landmarks)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Manos", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para salir
            break

        if key == ord("c"):
            # Cambiar de cámara (0 <-> 1)
            cap.release()
            cam_index = 1 - cam_index
            cap = open_camera(cam_index)
            if cap is None:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
