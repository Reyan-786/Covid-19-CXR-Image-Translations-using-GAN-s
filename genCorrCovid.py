import numpy as np
import os
from os import listdir
from os.path import join
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot as plt
from PIL import Image


def load_images(path, size=(256, 256)):
    images_list = listdir(path)
    loaded_images = []
    for image_name in images_list:
        img_path = join(path, image_name)
        image = load_img(img_path, target_size=size)
        image = img_to_array(image)
        loaded_images.append(image)
    return np.asarray(loaded_images)

def preprocess_images(images):
    images = (images - 127.5) / 127.5
    return images

def save_images_side_by_side(original, generated, index, save_dir='Results/'):
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert the images to unsigned bytes for PIL compatibility
    original_uint8 = ((original * 0.5 + 0.5) * 255).astype(np.uint8)
    generated_uint8 = ((generated[0] * 0.5 + 0.5) * 255).astype(np.uint8)
    
    # Create PIL Image objects
    original_img = Image.fromarray(original_uint8)
    generated_img = Image.fromarray(generated_uint8)
    
    # Combine the images side by side
    dst = Image.new('RGB', (original_img.width + generated_img.width, original_img.height))
    dst.paste(original_img, (0, 0))
    dst.paste(generated_img, (original_img.width, 0))
    
    # Save the combined image
    combined_path = os.path.join(save_dir, f'combined_{index}.png')
    dst.save(combined_path)

# Model custom objects
cust = {'InstanceNormalization': InstanceNormalization}

# Load models (Specify the paths to your trained models)
model_AtoB = load_model('Models/g_model_AtoB15000.h5', cust)

# Path to the dataset
path_normal = './Dataset1/Normal/'

# Load and preprocess all images from the Normal category
dataA_all = load_images(path_normal)
print('Loaded dataA: ', dataA_all.shape)
dataA_all = preprocess_images(dataA_all)

# Generate and save Covid-19 versions for each Normal image
for i in range(dataA_all.shape[0]):
    A_real = np.expand_dims(dataA_all[i], axis=0)  # Add batch dimension
    B_generated = model_AtoB.predict(A_real)
    
    # Save the original and generated images
    save_images_side_by_side(A_real[0], B_generated, i)

print(f"All images saved in './generated/' directory.")
