import cv2
import numpy as np
import matplotlib.pyplot as plt

ruta_imagen = 'imagenes/ardilla.jpg'

# ==========================================================================
#                 CÓDIGO APLICACIÓN I - Transformaciones
# ==========================================================================

# --------------------------------------------------
# ESCALAR IMAGEN 
def escalar_imagen(ruta_imagen, factor_x, factor_y):
    imagen = cv2.imread(ruta_imagen)

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    alto, ancho, _ = imagen.shape

    escalada_alto = int(alto * factor_y)
    escalada_ancho = int(ancho * factor_x)

    matriz_escalado = np.array([[factor_x, 0, 0], [0, factor_y, 0]])

    imagen_escalada = cv2.warpAffine(imagen_rgb, matriz_escalado, (escalada_ancho, escalada_alto))

    nuevo_alto, nuevo_ancho, _ = imagen_escalada.shape

    plt.figure(figsize=(10, 5),num=f'Aplicacion I - Escalar Imagen')

    plt.subplot(1, 2, 1)
    plt.title(f'Imagen Original\nTamaño {alto}x{ancho}px')
    plt.imshow(imagen_rgb)

    plt.subplot(1, 2, 2)
    plt.title(f'Imagen Escalada (Factor X: {factor_x}, Factor Y: {factor_y})\nTamaño {nuevo_alto}x{nuevo_ancho}px')
    plt.imshow(imagen_escalada)
    
    plt.show()
    return

# ---------------------------------------------
# ROTAR IMAGEN 
def rotar_imagen(ruta_imagen, angulo_rotacion):
    imagen = cv2.imread(ruta_imagen)

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    alto, ancho, _ = imagen.shape

    centro = (ancho // 2, alto // 2)

    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo_rotacion, 1.0)

    imagen_rotada = cv2.warpAffine(imagen_rgb, matriz_rotacion, (ancho, alto))

    plt.figure(figsize=(10, 5),num=f'Aplicacion I - Rotar Imagen')

    plt.subplot(1, 2, 1)
    plt.title('Imagen Original')
    plt.imshow(imagen_rgb)
    
    plt.subplot(1, 2, 2)
    plt.title(f'Imagen Rotada\n(Ángulo: {angulo_rotacion} grados)')
    plt.imshow(imagen_rotada)

    plt.show()
    return

# ---------------------------------------------------
# DEFORMAR IMAGEN 
def deformar_imagen(ruta_imagen, factor_x, factor_y):
    imagen = cv2.imread(ruta_imagen)

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    alto, ancho, _ = imagen.shape

    matriz_deformado = np.array([[1, factor_x, 0], [factor_y, 1, 0]])

    imagen_deformada = cv2.warpAffine(imagen_rgb, matriz_deformado, (ancho, alto))

    plt.figure(figsize=(10, 5),num=f'Aplicacion I - Deformar Imagen')

    plt.subplot(1, 2, 1)
    plt.title('Imagen Original')
    plt.imshow(imagen_rgb)
   
    plt.subplot(1, 2, 2)
    plt.title(f'Imagen Deformada\n(Factor X: {factor_x}, Factor Y: {factor_y})')
    plt.imshow(imagen_deformada)

    plt.show()
    return

# ==========================================================================
#               CÓDIGO APLICACIÓN II - Compresión de Imagenes
# ==========================================================================

# -----------------------------------
# COMPRIMIR IMAGEN
def comprimir_imagen(ruta_imagen, k):
    imagen = cv2.imread(ruta_imagen)
    imagen_original = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    imagen_para_comprimir = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)

    U, S, VT = np.linalg.svd(imagen_para_comprimir[:, :, 0], full_matrices=False)  # Canal Rojo
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    canal_rojo_comprimido = np.dot(U_k, np.dot(S_k, VT_k))

    U, S, VT = np.linalg.svd(imagen_para_comprimir[:, :, 1], full_matrices=False)  # Canal Verde
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    canal_verde_comprimido = np.dot(U_k, np.dot(S_k, VT_k))

    U, S, VT = np.linalg.svd(imagen_para_comprimir[:, :, 2], full_matrices=False)  # Canal Azul
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    canal_azul_comprimido = np.dot(U_k, np.dot(S_k, VT_k))

    imagen_comprimida = np.stack([canal_rojo_comprimido, canal_verde_comprimido, canal_azul_comprimido], axis=-1)
    imagen_comprimida = cv2.cvtColor(imagen_comprimida.astype(np.uint8), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5), num=f'Aplicacion II - Comprimir Imagen')

    plt.subplot(1, 2, 1)
    plt.title('Imagen Original')
    plt.imshow(imagen_original)

    plt.subplot(1, 2, 2)
    plt.title(f'Imagen Comprimida\n(k={k})')
    plt.imshow(imagen_comprimida)

    plt.show()
    return


# ==========================================================================
#                 EJECUTAR APLICACIÓN I - Transformaciones
# ==========================================================================
"""
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
"""

# ==========================================================================
#              EJECUTAR APLICACIÓN II - Compresión de Imagenes
# ==========================================================================

# COMPRIMIR IMAGEN
comprimir_imagen(ruta_imagen, 10)
comprimir_imagen(ruta_imagen, 50)
comprimir_imagen(ruta_imagen, 70)
comprimir_imagen(ruta_imagen, 100)