import cv2
import numpy as np
import matplotlib.pyplot as plt

def escalar_imagen(ruta_imagen, factor_x, factor_y):
    imagen = cv2.imread(ruta_imagen)

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    alto, ancho, _ = imagen.shape

    scaling_matrix = np.array([[factor_x, 0, 0], [0, factor_y, 0]])

    imagen_escalada = cv2.warpAffine(imagen_rgb, scaling_matrix, (ancho, alto))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Imagen Original')
    plt.imshow(imagen_rgb)

    plt.subplot(1, 2, 2)
    plt.title(f'Imagen Escalada (Factor X: {factor_x}, Factor Y: {factor_y})')
    plt.imshow(imagen_escalada)
    
    plt.show()

def rotar_imagen(ruta_imagen, angulo_rotacion):
    imagen = cv2.imread(ruta_imagen)

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    alto, ancho, _ = imagen.shape

    centro = (ancho // 2, alto // 2)

    rotation_matrix = cv2.getRotationMatrix2D(centro, angulo_rotacion, 1.0)

    imagen_rotada = cv2.warpAffine(imagen_rgb, rotation_matrix, (ancho, alto))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Imagen Original')
    plt.imshow(imagen_rgb)
    
    plt.subplot(1, 2, 2)
    plt.title(f'Imagen Rotada\n(Ángulo: {angulo_rotacion} grados)')
    plt.imshow(imagen_rotada)

    plt.show()


def deformar_imagen(ruta_imagen, factor_x, factor_y):
    imagen = cv2.imread(ruta_imagen)

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    alto, ancho, _ = imagen.shape

    shear_matrix = np.array([[1, factor_x, 0], [factor_y, 1, 0]])

    imagen_deformada = cv2.warpAffine(imagen_rgb, shear_matrix, (ancho, alto))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Imagen Original')
    plt.imshow(imagen_rgb)
   
    plt.subplot(1, 2, 2)
    plt.title(f'Imagen Deformada\n(Shear X: {factor_x}, Shear Y: {factor_y})')
    plt.imshow(imagen_deformada)

    plt.show()


ruta_imagen = 'imagenes/ardilla.jpg' # Cambia a la ruta de tu imagen

# ROTAR
angulo_rotacion = 45
rotar_imagen(ruta_imagen, angulo_rotacion)

angulo_rotacion = 69
rotar_imagen(ruta_imagen, angulo_rotacion)


# ESCALAR
escala_factor_x = 2.0
escala_factor_y = 1.5
escalar_imagen(ruta_imagen, escala_factor_x, escala_factor_y)

escala_factor_x = 3.0
escala_factor_y = 4.5
escalar_imagen(ruta_imagen, escala_factor_x, escala_factor_y)


# DEFORMAR
deformar_factor_x = 0.2
deformar_factor_y = 0.5
deformar_imagen(ruta_imagen, deformar_factor_x, deformar_factor_y)

deformar_factor_x = 0.1
deformar_factor_y = 0.9
deformar_imagen(ruta_imagen, deformar_factor_x, deformar_factor_y)