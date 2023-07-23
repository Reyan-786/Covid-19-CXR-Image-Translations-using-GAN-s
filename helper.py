from numpy.random import randint 
from numpy import vstack
from matplotlib import pyplot 
import numpy as np
from PIL import Image
import io
import numpy as np
from flask import send_file
from keras.utils import img_to_array
import tensorflow as tf
def select_sample(dataset, n_samples):
    	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X

# plot the image, its translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
	images = vstack((imagesX, imagesY1, imagesY2))
	titles = ['Real', 'Generated', 'Reconstructed']
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, len(images), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# title
		pyplot.title(titles[i])
	pyplot.show()



def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.LANCZOS)
    img = img_to_array(img)
    img /= 255.0  # Normalize pixel values to [0, 1]
    tensor = tf.constant(img, dtype=np.float32)
    tensor = tf.expand_dims(tensor, axis=0)
    return tensor


def load_model(path, custom_objects):
    return tf.keras.models.load_model(path, custom_objects=custom_objects)

def tensor_to_image(tensor):
    # Convert the TensorFlow tensor to a PIL image
    tensor = tf.squeeze(tensor, axis=0)
    tensor = (tensor+1) * 127.5
    tensor = tensor.numpy().astype('uint8')
    image = Image.fromarray(tensor)
    return image

def serve_pil_image(pil_image):
    # Serve the PIL image as a response
    img_io = io.BytesIO()
    pil_image.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')
