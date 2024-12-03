import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Görüntüyü yükle
image_path = 'datasets/Datos/train/goruntu2.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# LBP parametreleri
radius = 1  # LBP yarıçapı
n_points = 8 * radius  # Komşu sayısı

# LBP uygulama
lbp = local_binary_pattern(image, n_points, radius, method='uniform')

# LBP histogramını hesaplama
(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
hist = hist.astype("float")
hist /= (hist.sum() + 1e-6)  # Normalize histogram

# Orijinal ve LBP görüntüsünü gösterme
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Orijinal görüntü
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Orijinal Görüntü')
ax[0].axis('off')

# LBP görüntüsü
ax[1].imshow(lbp, cmap='gray')
ax[1].set_title('LBP Görüntüsü')
ax[1].axis('off')

plt.show()
