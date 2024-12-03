import numpy as np
import os
import cv2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Load the InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False)
layer_outputs = [layer.output for layer in base_model.layers]  # Extract outputs of all layers
activation_model = Model(inputs=base_model.input, outputs=layer_outputs)  # Create a model that will return these outputs

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image_resized = cv2.resize(image, (299, 299))  # Resize image to the required size for InceptionV3
    image_preprocessed = preprocess_input(image_resized)  # Preprocess image
    image_preprocessed = np.expand_dims(image_preprocessed, axis=0)  # Add batch dimension
    return image, image_preprocessed

def visualize_layer_outputs(layer_outputs, layer_names):
    images_per_row = 16
    for layer_name, layer_output in zip(layer_names, layer_outputs):
        n_features = layer_output.shape[-1]  # Number of features in the feature map
        size = layer_output.shape[1]  # Feature map size (height and width)
        n_cols = n_features // images_per_row  # Number of columns to display the images

        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_output[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

def main(image_path):
    image, preprocessed_image = preprocess_image(image_path)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    layer_outputs = activation_model.predict(preprocessed_image)
    layer_names = [layer.name for layer in base_model.layers]
    visualize_layer_outputs(layer_outputs[:12], layer_names[:12])  # Visualize first 12 layers for brevity

if __name__ == "__main__":
    image_path = 'datasets/Datos/train/goruntu2.png'
    main(image_path)
