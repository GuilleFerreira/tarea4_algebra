import cv2
import numpy as np
import matplotlib.pyplot as plt

def comprimir_imagen(ruta_imagen, k):
    imagen_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

    U, S, VT = np.linalg.svd(imagen_original, full_matrices=False)

    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]

    imagen_comprimida = np.dot(U_k, np.dot(S_k, VT_k))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Imagen Original')
    plt.imshow(imagen_original, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title(f'Imagen Comprimida (k={k})')
    plt.imshow(imagen_comprimida, cmap='gray')

    plt.show()


ruta_imagen = 'imagenes/ardilla.jpg'

comprimir_imagen(ruta_imagen, 10)
comprimir_imagen(ruta_imagen, 50)
comprimir_imagen(ruta_imagen, 70)
comprimir_imagen(ruta_imagen, 100)