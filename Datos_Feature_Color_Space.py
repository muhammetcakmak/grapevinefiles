import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image_path = 'datasets/Datos/train/goruntu.png'
image = cv2.imread(image_path)

# Color-Space özellik çıkarımı için örnek uygulama (HSV ve LAB dönüşümleri)
def color_space_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return hsv_image, lab_image

# HSV ve LAB dönüşümleri
hsv_image, lab_image = color_space_features(image)

# Görüntüleri çizimle göster
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Orijinal Görüntü')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
axes[1].set_title('HSV Transform')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB))
axes[2].set_title('LAB Transform')
axes[2].axis('off')

plt.show()
