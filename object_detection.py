import cv2
import numpy as np
from pynput.mouse import Listener as MouseListener

# Rangos HSV para el color verde fosforescente
verdeBajo = np.array([35, 100, 100], np.uint8)
verdeAlto = np.array([85, 255, 255], np.uint8)

# Rangos HSV para el color morado
moradoBajo = np.array([125, 100, 100], np.uint8)
moradoAlto = np.array([150, 255, 255], np.uint8)

# Estado para controlar el modo de visualización
modo_visualizacion = 0  # 0 = Primer paso, 1 = Segundo paso, 2 = Tercer paso, 3 = Cuarto paso

# Función para manejar los clics del mouse
def on_click(x, y, button, pressed):
    global modo_visualizacion
    if pressed:
        modo_visualizacion = (modo_visualizacion + 1) % 30  # Ciclar entre 0 y 29

# Iniciar el oyente del mouse en un hilo aparte
listener = MouseListener(on_click=on_click)
listener.start()

def detect_objects(frame):
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Crear máscaras para los colores verde y morado
    maskVerde = cv2.inRange(frameHSV, verdeBajo, verdeAlto)
    maskMorado = cv2.inRange(frameHSV, moradoBajo, moradoAlto)
    
    # Encontrar contornos verdes
    contornosVerdes, _ = cv2.findContours(maskVerde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Encontrar contornos morados (línea morada)
    contornosMorados, _ = cv2.findContours(maskMorado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    overlay = frame.copy()
    
    # Detectar el borde superior usando la línea morada y cambiar su color a azul
    borde_superior_y = frame.shape[0]  # Inicialmente, establecerlo al valor más bajo posible
    for contorno in contornosMorados:
        area = cv2.contourArea(contorno)
        if area > 500:  # Filtrar contornos pequeños
            x, y, w, h = cv2.boundingRect(contorno)
            borde_superior_y = min(borde_superior_y, y)  # Buscar el contorno más alto
            cv2.drawContours(frame, [contorno], 0, (255, 0, 0), thickness=cv2.FILLED)  # Cambiar color a azul

    for contorno in contornosVerdes:
        area = cv2.contourArea(contorno)
        if area > 500:  # Filtrar contornos pequeños
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            
            # Dividir el área verde en tres fragmentos
            x, y, w, h = cv2.boundingRect(contorno)
            fragmento_1 = (y, y + h // 3)
            fragmento_2 = (y + h // 3, y + 2 * h // 3)
            fragmento_3 = (y + 2 * h // 3, y + h)

            # Dividir el área verde en dos secciones verticalmente
            seccion_1 = (x, x + w // 3)
            seccion_2 = (x + w // 3, x + 2*w//3)
            seccion_3 = (x + 2*w // 3, x + w)


            # Dibujar el contorno y realizar operaciones basadas en el modo de visualización
            if modo_visualizacion == 0:
                cv2.drawContours(overlay, [contorno], 0, (0, 0, 255), thickness=cv2.FILLED)
                cv2.circle(overlay, (cX, cY), 20, (255, 255, 255), -1)
                cv2.putText(overlay, "Cubrir el circulo con gel", (cX-100, cY-100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            elif modo_visualizacion == 1:
                epsilon = 0.01 * cv2.arcLength(contorno, True)
                approx = cv2.approxPolyDP(contorno, epsilon, True)
                points = approx.squeeze()
                selected_points = points[np.round(np.linspace(0, len(points) - 1, 6)).astype(int)]
                for (x, y) in selected_points:
                    cv2.circle(overlay, (x, y), 20, (0, 0, 255), -1)
                cv2.putText(overlay, "DISTRIBUIR UNIFORMEMENTE EL GEL EN LA ZONA DESEADA", (10, borde_superior_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif modo_visualizacion == 2:
                cv2.putText(overlay, "COLOQUE EL NOTCH EN LA PARTE DERECHA DEL PACIENTE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            elif modo_visualizacion == 3:
                arco_x = x + 70
                arco_y = fragmento_1[0] + 30
                cv2.ellipse(overlay, (arco_x, arco_y), (30, 10), 0, 180, 360, (0, 0, 0), 4)
                cv2.putText(overlay, "ARCO", (arco_x - 50, arco_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif modo_visualizacion == 4:
                linea1_y = fragmento_1[0] + (fragmento_1[1] - fragmento_1[0]) // 2
                linea2_y = fragmento_2[0] + (fragmento_2[1] - fragmento_2[0]) // 2
                for i in range(x, x + w, 20):
                    cv2.line(overlay, (i, linea1_y), (i + 10, linea1_y), (0, 0, 0), 2)
                for i in range(x, x + w, 20):
                    cv2.line(overlay, (i, linea2_y), (i + 10, linea2_y), (0, 0, 0), 2)
                cv2.putText(overlay, "DESLIZA", (x + 10, linea1_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            elif modo_visualizacion == 5:
                arco_radio = 30
                center_x = x + w - 70
                center_y = fragmento_1[0] + 30
                cv2.ellipse(overlay, (center_x, center_y), (arco_radio, 10), 0, 180, 360, (0, 0, 0), 4)
                cv2.putText(overlay, "ARCO", (center_x - 50, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif modo_visualizacion == 6:
                cv2.putText(overlay, "COLOQUE EL TRANSDUCTOR DEBAJO DE LAS COSTILLAS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            elif modo_visualizacion == 7:
                arco_radio = 30
                arco_x = x + w // 2
                arco_y = fragmento_1[1] + arco_radio
                cv2.ellipse(overlay, (arco_x, arco_y), (arco_radio, 10), 0, 180, 360, (0, 0, 0), 4)
                cv2.putText(overlay, "ARCO EN BORDE INFERIOR", (arco_x - 50, arco_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif modo_visualizacion == 8:
                linea2_y = fragmento_2[0] + (fragmento_2[1] - fragmento_2[0]) // 2
                linea3_y = fragmento_3[0] + (fragmento_3[1] - fragmento_3[0]) // 2
                for i in range(x, x + w, 20):
                    cv2.line(overlay, (i, linea2_y), (i + 10, linea2_y), (0, 0, 0), 2)
                for i in range(x, x + w, 20):
                    cv2.line(overlay, (i, linea3_y), (i + 10, linea3_y), (0, 0, 0), 2)
                cv2.putText(overlay, "DESLIZA", (x + 10, linea2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            elif modo_visualizacion == 9:
                arco_radio = 30
                center_x = x + w - 70
                center_y = (fragmento_2[0] + 30)
                cv2.ellipse(overlay, (center_x, center_y), (arco_radio, 10), 0, 180, 360, (0, 0, 0), 4)
                cv2.putText(overlay, "ARCO EN BORDE DERECHO", (center_x - 50, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif modo_visualizacion == 10:
                cv2.putText(overlay, "COLOQUE EL TRANSDUCTOR DEBAJO DE LAS COSTILLAS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            elif modo_visualizacion == 11:
                arco_radio = 20
                arco_x = x + w // 2
                arco_y = fragmento_2[1] + arco_radio
                cv2.ellipse(overlay, (arco_x, arco_y), (arco_radio, 10), 0, 180, 360, (0, 0, 0), 4)
                cv2.putText(overlay, "ARCO", (arco_x - 50, arco_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif modo_visualizacion == 12:
                linea2_y = fragmento_3[0]
                linea3_y = fragmento_3[1]
                for i in range(x, x + w, 20):
                    cv2.line(overlay, (i, linea2_y), (i + 10, linea2_y), (0, 0, 0), 2)
                for i in range(x, x + w, 20):
                    cv2.line(overlay, (i, linea3_y), (i + 10, linea3_y), (0, 0, 0), 2)
                cv2.putText(overlay, "DESLIZA", (x + 10, linea2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            elif modo_visualizacion == 13:
                arco_radio = 20
                center_x = x + w - 70
                center_y = fragmento_2[1] + arco_radio
                cv2.ellipse(overlay, (center_x, center_y), (arco_radio, 10), 0, 180, 360, (0, 0, 0), 4)
                cv2.putText(overlay, "ARCO EN BORDE DERECHO", (center_x - 50, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif modo_visualizacion == 14:
                cv2.putText(overlay, "INDICAR AL PACIENTE QUE GIRE 45° A LA IZQUIERDA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            elif modo_visualizacion == 15:
                arco_radio = 30
                arco_x = x + w // 2
                arco_y = fragmento_1[1] + arco_radio
                cv2.ellipse(overlay, (arco_x, arco_y), (arco_radio, 10), 0, 180, 360, (0, 0, 0), 4)
                cv2.putText(overlay, "ARCO EN BORDE INFERIOR", (arco_x - 50, arco_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif modo_visualizacion == 16:
                linea2_y = fragmento_2[0] + (fragmento_2[1] - fragmento_2[0]) // 2
                linea3_y = fragmento_3[0] + (fragmento_3[1] - fragmento_3[0]) // 2
                for i in range(x, x + w, 20):
                    cv2.line(overlay, (i, linea2_y), (i + 10, linea2_y), (0, 0, 0), 2)
                for i in range(x, x + w, 20):
                    cv2.line(overlay, (i, linea3_y), (i + 10, linea3_y), (0, 0, 0), 2)
                cv2.putText(overlay, "DESLIZA", (x + 10, linea2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            elif modo_visualizacion == 17:
                arco_radio = 30
                center_x = x + w - 70
                center_y = (fragmento_2[0] + 30)
                cv2.ellipse(overlay, (center_x, center_y), (arco_radio, 10), 0, 180, 360, (0, 0, 0), 4)
                cv2.putText(overlay, "ARCO", (center_x - 50, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif modo_visualizacion == 18:
                topmost = tuple(contorno[contorno[:, :, 1].argmin()][0])
                bbox = cv2.boundingRect(contorno)
                center_y = bbox[0] + bbox[2] // 3
                center_x = topmost[1] + 50
                cv2.ellipse(overlay, (center_y, center_x), (30, 10), 270, 0, 180, (0, 0, 0), 4)
                cv2.putText(overlay, "ARCO", (center_y - 50, center_x - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif modo_visualizacion == 19:
                cv2.putText(overlay, "INDICAR AL PACIENTE QUE SE PONGA DE PIE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            elif modo_visualizacion == 20:
                arco_radio = 20
                arco_x = x + w // 2
                arco_y = fragmento_1[0] + 20
                cv2.ellipse(overlay, (arco_x, arco_y), (arco_radio, 7), 0, 180, 360, (0, 0, 0), 4)
                cv2.putText(overlay, "ARCO", (arco_x - 20, arco_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            elif modo_visualizacion == 21:
                linea1_x = seccion_2[0]
                linea2_x = seccion_2[1]
                linea_y_start = fragmento_1[0]
                linea_y_end = fragmento_3[1]
                for i in range(linea_y_start, linea_y_end, 20):
                    cv2.line(overlay, (linea1_x, i), (linea1_x, i + 10), (0, 0, 0), 2)
                cv2.putText(overlay, "DESLIZA", (linea1_x - 30, linea_y_start - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                for i in range(linea_y_start, linea_y_end, 20):
                    cv2.line(overlay, (linea2_x, i), (linea2_x, i + 10), (0, 0, 0), 2)

            elif modo_visualizacion == 22:
                arco_radio = 20
                arco_x = x + w // 2
                arco_y = fragmento_3[0] + (fragmento_3[1] - fragmento_3[0]) // 2
                cv2.ellipse(overlay, (arco_x, arco_y), (arco_radio, 7), 0, 180, 360, (0, 0, 0), 4)
                cv2.putText(overlay, "ARCO", (arco_x - 20, arco_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            elif modo_visualizacion == 23:
                cv2.putText(overlay, "HA CULMINADO CON LOS PASOS DE PROTOCOLO VSI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame
