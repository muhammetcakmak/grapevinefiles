import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops

# Load the image
image_path = 'datasets/Datos/train/goruntu2.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to calculate GLCM features
def calculate_glcm_features(image):
    glcm = greycomatrix(image, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    asm = greycoprops(glcm, 'ASM')[0, 0]
    return contrast, dissimilarity, homogeneity, energy, correlation, asm

# Calculate GLCM features
contrast, dissimilarity, homogeneity, energy, correlation, asm = calculate_glcm_features(image)

# Visualize the original image and GLCM features
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

# Display GLCM features
ax[1].text(0.1, 0.8, f'Contrast: {contrast:.2f}', fontsize=12)
ax[1].text(0.1, 0.7, f'Dissimilarity: {dissimilarity:.2f}', fontsize=12)
ax[1].text(0.1, 0.6, f'Homogeneity: {homogeneity:.2f}', fontsize=12)
ax[1].text(0.1, 0.5, f'Energy: {energy:.2f}', fontsize=12)
ax[1].text(0.1, 0.4, f'Correlation: {correlation:.2f}', fontsize=12)
ax[1].text(0.1, 0.3, f'ASM: {asm:.2f}', fontsize=12)
ax[1].set_title('GLCM Features')
ax[1].axis('off')

plt.show()
